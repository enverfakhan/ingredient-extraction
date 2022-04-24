from __future__ import absolute_import, division, print_function
import logging
import os
import random
import string
import tqdm
import pandas as pd
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)


from utils.foodstyle_utils import replace_contractions
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



class InputExample:
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures:
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask


def readfile(filename):
    '''
    read file
    '''
    f = open(filename)
    data = []
    sentence = []
    label = []
    for line in f:
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            if len(sentence) > 0:
                data.append((sentence, label))
                sentence = []
                label = []
            continue
        splits = line.split(' ')
        sentence.append(splits[0])
        label.append(splits[-1][:-1])

    if len(sentence) > 0:
        data.append((sentence, label))
        sentence = []
        label = []
    return data


def recipe_to_sentences(recipe, ingrs):
    lemmatizer = WordNetLemmatizer()
    label_map = {k.lower(): 'ING' for ingr in ingrs for k in ingr.split(' ')}
    steps = recipe.split('**')
    data = []
    for step in steps:
        if step == '':
            continue
        sentences = step.split('.')
        for sentence in sentences:
            if sentence == '':
                continue
            sentence = replace_contractions(sentence)  # replacing contractions
            sentence = sentence.replace("'s", " 's")  # replacing apostrophes
            sentence = sentence.split(' ')
            label_seq = []
            new_sentence = []
            for word in sentence:
                if word == '':
                    continue
                new_sentence.append(word)
                word = word.translate(str.maketrans('', '', string.punctuation))
                word = lemmatizer.lemmatize(word)
                label = label_map.get(word.lower(), 'N-ING')
                label_seq.append(label)
            if len(new_sentence) > 0:
                new_sentence[-1] = new_sentence[-1] + '.'
            data.append((new_sentence, label_seq))

    return data


def readcsvfile(filename):
    dataframe = pd.read_csv(filename, on_bad_lines='skip', delimiter=';')
    dataframe = dataframe.dropna(subset=['Ingredients', 'Directions'])
    ingredients = [d.split(',') for d in dataframe['Ingredients']]
    directions = list(dataframe['Directions'].values)
    samples = list(zip(directions, ingredients))
    random.seed(42)
    eval_indices = set(random.choices(range(len(samples)), k=int(0.15 * len(samples))))
    eval_data = []
    train_data = []
    cnt = 0
    for i, (recipe, ingrs) in enumerate(samples):
        # open below for debugging
        # if cnt > 1000:
        #     break
        parsed_recipe = recipe_to_sentences(recipe, ingrs)
        if i in eval_indices:
            eval_data.extend(parsed_recipe)
        else:
            train_data.extend(parsed_recipe)
        cnt += 1

    return train_data, eval_data


class DataProcessor:
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file):
        """Reads a tab separated value file."""
        return readfile(input_file)

    @classmethod
    def _read_csv(cls, input_file):
        """Reads a tab separated value file."""
        return readcsvfile(input_file)


class NerProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def __init__(self):
        self.train = None
        self.dev = None
        self.test = None

    def create_sets(self, data_dir):
        train_lines, eval_lines = self._read_csv(os.path.join(data_dir, "recipes.csv"))
        print(f"number of sentences in train set: {len(train_lines)}")
        print(f"number of sentences in train set: {len(eval_lines)}")

        train_examples = self._create_examples(train_lines, "train")

        eval_examples = self._create_examples(eval_lines, "dev")
        self.train = train_examples
        self.dev = eval_examples

    def get_train_examples(self, data_dir):
        """See base class."""
        if self.train is None:
            self.create_sets(data_dir)

        return self.train

    def get_dev_examples(self, data_dir):
        """See base class."""
        if self.dev is None:
            self.create_sets(data_dir)

        return self.dev

    def get_test_examples(self, data_dir):
        """See base class."""
        if self.dev is None:
            self.create_sets(data_dir)

        return self.dev

    def get_labels(self):
        # return ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]", "[SEP]"]
        return ["N-ING", "ING", "[CLS]", "[SEP]"]

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(
                guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list, 1)}

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples)):
        textlist = example.text_a.split(' ')
        labellist = example.label
        tokens = []
        labels = []
        valid = []
        label_mask = []
        start_position = 1
        for i, word in enumerate(textlist):
            if i >= len(labellist):
                continue
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            labels.append(label_1)
            valid.append(start_position)
            start_position += len(token)
            label_mask.append(True)
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
            label_mask = label_mask[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid.insert(0, 0)
        label_mask.insert(0, True)
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                label_ids.append(label_map[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid.append(valid[-1]+1)
        label_mask.append(True)
        label_ids.append(label_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        label_mask = [True] * len(label_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            label_mask.append(False)
        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(False)
        while len(valid) < max_seq_length:
            valid.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" %
                        " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" %
                        " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_ids,
                          valid_ids=valid,
                          label_mask=label_mask))
    return features

