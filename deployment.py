from model import BertNer
import json
import os
from tokenization import FullTokenizer
from utils.foodstyle_utils import convert_logit_to_labels, get_ingredients_and_positions
from prepare_data import NerProcessor, convert_examples_to_features, recipe_to_sentences
import numpy as np

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)


class IngredientExtractioonInference:

    def __init__(self, path_to_model_dir):
        self.tokenizer = FullTokenizer(os.path.join(path_to_model_dir, "vocab.txt"), do_lower_case=True)

        self.max_seq_length = 50
        self.processor = NerProcessor()
        self.label_list = self.processor.get_labels()
        num_labels = len(self.label_list) + 1
        self.label_map = {i: label for i, label in enumerate(self.label_list, 1)}

        config = json.load(open(os.path.join('output/2022-4-24-14-21-9', "bert_config.json")))
        self.model = BertNer(config, tf.float32, num_labels, self.max_seq_length)

        ids = tf.ones((1, 50), dtype=tf.int32)
        _ = self.model(ids, ids, ids, ids, training=False)
        self.model.load_weights(os.path.join(path_to_model_dir, "model-8.h5"))

    def __call__(self, recipes):
        result = {}
        for recipe_name, sample_recipe in recipes.items():
            # step-1 create features (tokenize)
            sentences_labels_pair = recipe_to_sentences(sample_recipe, ingrs=[])
            sample_example = self.processor._create_examples(sentences_labels_pair, 'test')
            sample_features = convert_examples_to_features(sample_example, self.label_list,
                                                           self.max_seq_length, self.tokenizer)

            # step-2 prepare tensors for forward pass
            all_input_ids = tf.convert_to_tensor(np.asarray([f.input_ids for f in sample_features], dtype=np.int32))
            all_input_mask = tf.convert_to_tensor(
                np.asarray([f.input_mask for f in sample_features], dtype=np.int32))
            all_segment_ids = tf.convert_to_tensor(
                np.asarray([f.segment_ids for f in sample_features], dtype=np.int32))
            all_valid_ids = tf.convert_to_tensor(
                np.asarray([f.valid_ids for f in sample_features], dtype=np.int32))
            all_label_ids = tf.convert_to_tensor(
                np.asarray([f.label_id for f in sample_features], dtype=np.int32))


            # step-3 forwad pass the model and get the logits
            logits = self.model(all_input_ids, all_input_mask, all_segment_ids, all_valid_ids, training=False)


            # step-4 conver model's output (logits) to predictions and tokens
            pred, tokens = convert_logit_to_labels(logits, all_label_ids, self.label_map, all_input_ids, self.tokenizer)

            # step-5 get ingredients and their positions from prediction and token pairs
            ing_and_positions = get_ingredients_and_positions(pred, tokens)
            result[recipe_name] = ing_and_positions

        return result


if __name__ == '__main__':
    inference = IngredientExtractioonInference('output/2022-4-24-14-21-9')
    sample_recipe = "In a large bowl, combine flour, baking powder, baking soda, salt, cinnamon, nutmeg, brown sugar, and  oats. Add apple, nuts, raisins, eggs, milk, and oil.  Mix until dry ingredients are moistened.**Bake for 55 to 60 minutes, or until done.  Cool on wire rack."
    recipes = {'recipe_name': sample_recipe}
    ingredients = inference(recipes)
    print('done')
