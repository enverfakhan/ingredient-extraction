from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import logging
import math
import os
import datetime
import random
import string
import shutil
import sys
import tqdm

import numpy as np
import pandas as pd
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
from fastprogress import master_bar, progress_bar
# from seqeval.metrics import classification_report
from sklearn.metrics import classification_report

from model import BertNer
from optimization import AdamWeightDecay, WarmUp
from tokenization import FullTokenizer

from utils.foodstyle_utils import replace_contractions, flatten_class_report
from prepare_data import NerProcessor, convert_examples_to_features

import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

try:
    import wandb
    has_wandb = True
except:
    has_wandb = False


def create_directory_name():
    now = datetime.datetime.now()
    now = '-'.join([str(now.year), str(now.month), str(now.day), str(now.hour), str(now.minute), str(now.second)])
    return f'output/{now}'


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default='cased_L-12_H-768_A-12', type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-cased,bert-large-cased")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # Other parameters
    parser.add_argument("--max_seq_length",
                        default=50,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        default=True,
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        default=True,
                        help="Whether to run eval on the dev/test set.")
    parser.add_argument("--eval_on",
                        default="dev",
                        type=str,
                        help="Evaluation set, dev: Development, test: Test")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=2,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=2,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    # training stratergy arguments
    parser.add_argument("--multi_gpu",
                        action='store_true',
                        help="Set this flag to enable multi-gpu training using MirroredStrategy."
                             "Single gpu training")
    parser.add_argument("--gpus",default='0',type=str,
                        help="Comma separated list of gpus devices."
                              "For Single gpu pass the gpu id.Default '0' GPU"
                              "For Multi gpu,if gpus not specified all the available gpus will be used")

    args = parser.parse_args()
    if has_wandb:
        wandb.init(project='ingredient-extraction', config=args, entity='envers-workshop')
    processor = NerProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list) + 1
    directory = create_directory_name()
    args.output_dir = directory
    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    if args.do_train:
        tokenizer = FullTokenizer(os.path.join(args.bert_model, "vocab.txt"), args.do_lower_case)

    if args.multi_gpu:
        if len(args.gpus.split(',')) == 1:
            strategy = tf.distribute.MirroredStrategy()
        else:
            gpus = [f"/gpu:{gpu}" for gpu in args.gpus.split(',')]
            strategy = tf.distribute.MirroredStrategy(devices=gpus)
    else:
        gpu = args.gpus.split(',')[0]
        strategy = tf.distribute.OneDeviceStrategy(device=f"/gpu:{gpu}")

    train_examples = None
    optimizer = None
    num_train_optimization_steps = 0
    ner = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size) * args.num_train_epochs
        warmup_steps = int(args.warmup_proportion *
                           num_train_optimization_steps)
        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=args.learning_rate,
                                                decay_steps=num_train_optimization_steps,end_learning_rate=0.0)
        if warmup_steps:
            learning_rate_fn = WarmUp(initial_learning_rate=args.learning_rate,
                                    decay_schedule_fn=learning_rate_fn,
                                    warmup_steps=warmup_steps)
        optimizer = AdamWeightDecay(
            learning_rate=learning_rate_fn,
            weight_decay_rate=args.weight_decay,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=args.adam_epsilon,
            exclude_from_weight_decay=['layer_norm', 'bias'])

        with strategy.scope():
            ner = BertNer(args.bert_model, tf.float32, num_labels, args.max_seq_length)
            loss_fct = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    label_map = {i: label for i, label in enumerate(label_list, 1)}
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        all_input_ids = tf.data.Dataset.from_tensor_slices(
            np.asarray([f.input_ids for f in train_features],dtype=np.int32))
        all_input_mask = tf.data.Dataset.from_tensor_slices(
            np.asarray([f.input_mask for f in train_features],dtype=np.int32))
        all_segment_ids = tf.data.Dataset.from_tensor_slices(
            np.asarray([f.segment_ids for f in train_features],dtype=np.int32))
        all_valid_ids = tf.data.Dataset.from_tensor_slices(
            np.asarray([f.valid_ids for f in train_features],dtype=np.int32))
        all_label_mask = tf.data.Dataset.from_tensor_slices(
            np.asarray([f.label_mask for f in train_features]))

        all_label_ids = tf.data.Dataset.from_tensor_slices(
            np.asarray([f.label_id for f in train_features],dtype=np.int32))

        # Dataset using tf.data
        train_data = tf.data.Dataset.zip(
            (all_input_ids, all_input_mask, all_segment_ids, all_valid_ids, all_label_ids,all_label_mask))
        shuffled_train_data = train_data.shuffle(buffer_size=int(len(train_features) * 0.1),
                                                seed = args.seed,
                                                reshuffle_each_iteration=True)
        batched_train_data = shuffled_train_data.batch(args.train_batch_size)
        # Distributed dataset
        dist_dataset = strategy.experimental_distribute_dataset(batched_train_data)
        # dist_dataset = batched_train_data

        loss_train_metric = tf.keras.metrics.Mean()
        loss_eval_metric = tf.keras.metrics.Mean()

        epoch_bar = master_bar(range(args.num_train_epochs))
        pb_max_len = math.ceil(
            float(len(train_features))/float(args.train_batch_size))

        def train_step(input_ids, input_mask, segment_ids, valid_ids, label_ids,label_mask):
            def step_fn(input_ids, input_mask, segment_ids, valid_ids, label_ids,label_mask):

                with tf.GradientTape() as tape:
                    try:
                        logits = ner(input_ids, input_mask,segment_ids, valid_ids, training=True)
                    except Exception as e:
                        return None
                    label_mask = tf.reshape(label_mask,(-1,))
                    logits = tf.reshape(logits,(-1,num_labels))
                    logits_masked = tf.boolean_mask(logits,label_mask)
                    label_ids = tf.reshape(label_ids,(-1,))
                    label_ids_masked = tf.boolean_mask(label_ids,label_mask)
                    cross_entropy = loss_fct(label_ids_masked, logits_masked)
                    loss = tf.reduce_sum(cross_entropy) * (1.0 / args.train_batch_size)
                grads = tape.gradient(loss, ner.trainable_variables)
                optimizer.apply_gradients(list(zip(grads, ner.trainable_variables)))
                return cross_entropy

            per_example_losses = strategy.run(step_fn, args=(input_ids, input_mask, segment_ids,
                                                                              valid_ids, label_ids,label_mask))
            if per_example_losses is None:
                return None
            mean_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_example_losses, axis=0)
            return mean_loss

        # copy vocab to output_dir
        shutil.copyfile(os.path.join(args.bert_model, "vocab.txt"), os.path.join(args.output_dir, "vocab.txt"))
        # copy bert config to output_dir
        shutil.copyfile(os.path.join(args.bert_model, "bert_config.json"),
                        os.path.join(args.output_dir, "bert_config.json"))
        # save label_map and max_seq_length of trained model
        model_config = {"bert_model": args.bert_model, "do_lower": args.do_lower_case,
                        "max_seq_length": args.max_seq_length, "num_labels": num_labels,
                        "label_map": label_map}
        json.dump(model_config, open(os.path.join(args.output_dir, "model_config.json"), "w"), indent=4)

        for epoch in epoch_bar:
            logger.info(f"epoch number: {epoch}")
            with strategy.scope():
               for (input_ids, input_mask, segment_ids, valid_ids, label_ids,label_mask) in progress_bar(dist_dataset, total=pb_max_len, parent=epoch_bar):
                   print(f"the epcoh number is: {epoch}")
                   loss = train_step(input_ids, input_mask, segment_ids, valid_ids, label_ids,label_mask)
                   if loss is None:
                       continue
                   loss_train_metric(loss)
                   epoch_bar.child.comment = f'loss : {loss_train_metric.result()}'
            
            # model wight save
            ner.save_weights(os.path.join(args.output_dir,f"model-{epoch}.h5"))


            if args.do_eval:
                # load tokenizer
                tokenizer = FullTokenizer(os.path.join(args.output_dir, "vocab.txt"), args.do_lower_case)
                # model build hack : fix
                # config = json.load(open(os.path.join(args.output_dir,"bert_config.json")))
                # ner = BertNer(config, tf.float32, num_labels, args.max_seq_length)
                ids = tf.ones((1,50),dtype=tf.int32)
                _ = ner(ids,ids,ids,ids, training=False)
                # ner.load_weights(os.path.join(args.output_dir,f"model-{epoch}.h5"))

                # load test or development set based on argsK
                if args.eval_on == "dev":
                    eval_examples = processor.get_dev_examples(args.data_dir)
                elif args.eval_on == "test":
                    eval_examples = processor.get_test_examples(args.data_dir)

                eval_features = convert_examples_to_features(
                    eval_examples, label_list, args.max_seq_length, tokenizer)
                logger.info("***** Running evalution *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)

                all_input_ids_eval = tf.data.Dataset.from_tensor_slices(
                    np.asarray([f.input_ids for f in eval_features],dtype=np.int32))
                all_input_mask_eval = tf.data.Dataset.from_tensor_slices(
                    np.asarray([f.input_mask for f in eval_features],dtype=np.int32))
                all_segment_ids_eval = tf.data.Dataset.from_tensor_slices(
                    np.asarray([f.segment_ids for f in eval_features],dtype=np.int32))
                all_valid_ids_eval = tf.data.Dataset.from_tensor_slices(
                    np.asarray([f.valid_ids for f in eval_features],dtype=np.int32))

                all_label_ids_eval = tf.data.Dataset.from_tensor_slices(
                    np.asarray([f.label_id for f in eval_features],dtype=np.int32))

                all_label_mask_eval = tf.data.Dataset.from_tensor_slices(
                    np.asarray([f.label_mask for f in eval_features]))

                eval_data = tf.data.Dataset.zip(
                    (all_input_ids_eval, all_input_mask_eval, all_segment_ids_eval, all_valid_ids_eval,
                     all_label_ids_eval, all_label_mask_eval))
                batched_eval_data = eval_data.batch(args.eval_batch_size)

                epoch_bar = master_bar(range(1))
                pb_max_len = math.ceil(
                    float(len(eval_features))/float(args.eval_batch_size))

                y_true = []
                y_pred = []
                label_map = {i : label for i, label in enumerate(label_list,1)}
                for epoch in epoch_bar:
                    for (input_ids, input_mask, segment_ids, valid_ids, label_ids_orig, label_mask) in progress_bar(batched_eval_data, total=pb_max_len, parent=epoch_bar):
                            logits = ner(input_ids, input_mask, segment_ids, valid_ids, training=False)
                            predictions = tf.argmax(logits, axis=2)
                            label_mask = tf.reshape(label_mask, (-1,))
                            logits = tf.reshape(logits, (-1, num_labels))
                            logits_masked = tf.boolean_mask(logits, label_mask)
                            label_ids = tf.reshape(label_ids_orig, (-1,))
                            label_ids_masked = tf.boolean_mask(label_ids, label_mask)
                            cross_entropy = loss_fct(label_ids_masked, logits_masked)
                            loss_eval_metric(cross_entropy)
                            for i, label in enumerate(label_ids_orig):
                                temp_1 = []
                                temp_2 = []
                                for j,m in enumerate(label):
                                    if j == 0:
                                        continue
                                    elif label_ids_orig[i][j].numpy() == len(label_map):
                                        y_true.extend(temp_1)
                                        y_pred.extend(temp_2)
                                        break
                                    else:
                                        temp_1.append(label_map[label_ids_orig[i][j].numpy()])
                                        temp_2.append(label_map[predictions[i][j].numpy()])
                metric_container = {'eval_loss': float(loss_eval_metric.result().numpy()),
                                    'train_loss': float(loss_train_metric.result().numpy())}
                loss_train_metric.reset_states()
                loss_eval_metric.reset_states()
                report_dict = classification_report(y_true, y_pred, digits=4, output_dict=True)
                report_dict = flatten_class_report(report_dict)
                for metric_name, metric in report_dict.items():
                    if metric.count == 0:
                        continue
                    metric_container[metric_name] = metric.value
                if has_wandb:
                    wandb.log(metric_container)
                print(metric_container.items(), flush=True)
                report = classification_report(y_true, y_pred, digits=4)
                output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
                with open(output_eval_file, "a") as writer:
                    logger.info("***** Eval results *****")
                    logger.info("\n%s", report)
                    writer.write(report)


if __name__ == "__main__":
    main()
