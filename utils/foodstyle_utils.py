import re
import string
from collections import namedtuple, Counter
import tensorflow as tf

### this part is taken from https://www.exxactcorp.com/blog/Deep-Learning/text-preprocessing-methods-for-deep-learning

contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because",
                    "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not",
                    "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                    "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you",
                    "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have",
                    "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                    "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am",
                    "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have",
                    "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                    "mayn't": "may not", "might've": "might have","mightn't": "might not",
                    "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
                    "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
                    "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
                    "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would",
                    "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                    "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have",
                    "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would",
                    "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                    "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would",
                    "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have",
                    "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not",
                    "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
                    "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will",
                    "what'll've": "what will have", "what're": "what are",  "what's": "what is",
                    "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did",
                    "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have",
                    "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have",
                    "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have",
                    "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                    "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are",
                    "y'all've": "you all have","you'd": "you would", "you'd've": "you would have",
                    "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"
                    }


def _get_contractions(contraction_dict):
    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    return contraction_re


contractions_re = _get_contractions(contraction_dict)


def replace_contractions(text):
    def replace(match):
        return contraction_dict[match.group(0)]
    return contractions_re.sub(replace, text)

## end of contractions replacement section


def flatten_class_report(report):
    flattened_dict_of_metrics = {}
    value_support = namedtuple('metric', ['value', 'count'])
    # print(report)
    for class_name, metric_dict in report.items():
        if type(metric_dict) is float:  # handle accuracy
            print(f'{class_name}: {metric_dict}')
            flattened_dict_of_metrics[class_name] = value_support._make([metric_dict, 1])
            continue
        for metric, value in metric_dict.items():
            if metric != 'support':
                flattened_metric_name = class_name + '_' + metric
                flattened_dict_of_metrics[flattened_metric_name] = value_support._make([value, metric_dict['support']])
    return flattened_dict_of_metrics


def get_sentence_len_histogram(data):
    distribution = Counter([len(sentence) for sentence in data])
    return distribution


def get_word_distribution(data):
    distribution = Counter()
    for sentence in data:
        for word in sentence:
            distribution.update(word)

    return distribution


def convert_logit_to_labels(logits, label_ids, label_map, input_ids, tokenizer):
    predictions = tf.argmax(logits, axis=2)
    y_pred = []
    tokens = []
    for i, label in enumerate(label_ids):
        temp_1 = []
        temp_2 = []
        for j,m in enumerate(label):
            if j == 0:
                continue
            token = tokenizer.convert_ids_to_tokens([input_ids[i][j].numpy()])[0]
            pred = label_map[predictions[i][j].numpy()]
            if token == '[PAD]':
                y_pred.extend(temp_2[:-1])
                tokens.extend(temp_1[:-1])
                break
            else:
                temp_2.append(pred)
                temp_1.append(token)
    return y_pred, tokens


def combine_phrases(result):
    new_result = [result[0]]
    for candid in result[1:]:
        if candid[1] == new_result[-1][2]:
            new_result[-1][0] = ' '.join([new_result[-1][0], candid[0]])
            new_result[-1][2] = candid[2]
        else:
            new_result.append(candid)
    return new_result


def get_ingredients_and_positions(predictions, tokens):
    start_position = 0
    end_position = 0
    new_word = False
    result = []
    word = ''
    is_ingr = False
    for i, (pred, token) in enumerate(zip(predictions, tokens)):
        if token in string.punctuation:
            start_position += 1
            continue
        is_ingr = is_ingr or pred == 'ING'
        if token[:2] == '##':
            end_position = end_position + len(token) -2
            word = word + token[2:]
            new_word = False
        else:
            if new_word is False:
                if is_ingr is True:
                    result.append([word, start_position-1, end_position])
                start_position = end_position
                word = ''
                is_ingr = False

            start_position += 1
            end_position = start_position + len(token)
            word = word + token
            if len(tokens) >= i and tokens[i+1][:2] == '##':
                new_word = False
            else:
                new_word = True
            is_ingr = is_ingr or pred == 'ING'
        if new_word is True:
            if is_ingr is True:
                result.append([word, start_position-1, end_position])
            start_position = end_position
            word = ''
            is_ingr = False

    result = combine_phrases(result)
    return result

