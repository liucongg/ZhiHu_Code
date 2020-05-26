"""
@author: liucong
@contact: logcongcong@gmail.com
@time: 2020/5/25 18:11
"""

import json
import numpy as np
from sklearn.datasets.base import Bunch
import bert_tokenization
import pickle


def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
        with open(filename, "w") as fh:
            json.dump(obj, fh)


def convert_features(config, content, tokenizer):
    max_seq_length = config.max_seq_length
    max_length_context = max_seq_length - 2

    content_token = tokenizer.tokenize(content)
    if len(content_token) > max_length_context:
        content_token = content_token[:max_length_context]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in content_token:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    input_ids = np.array(input_ids)
    input_mask = np.array(input_mask)
    segment_ids = np.array(segment_ids)
    return input_ids, input_mask, segment_ids


def build_features(config, file, data_type, tokenizer):
    label_list = ['体育', '娱乐', '家居', '彩票', '房产', '教育', '时尚', '时政', '星座', '游戏', '社会', '科技', '股票', '财经']
    bunch = Bunch(ids=[], input_ids=[], input_mask=[], segment_ids=[], label=[])
    print("Processing {} examples ...".format(data_type))
    with open(file, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            sample = json.loads(line.strip())
            news = sample["news"]
            label = label_list.index(sample["label"])
            one_hot = [0]*len(label_list)
            one_hot[label] = 1
            input_ids, input_mask, segment_ids = convert_features(config, news, tokenizer)
            bunch.input_ids.append(input_ids)
            bunch.input_mask.append(input_mask)
            bunch.segment_ids.append(segment_ids)
            bunch.label.append(one_hot)
    return bunch


def prepro(config):
    tokenizer = bert_tokenization.FullTokenizer(vocab_file=config.vocab_file)
    trainBunch = build_features(config, config.train_file, "train", tokenizer)
    print("save train bunch")
    pickle.dump(trainBunch, open(config.train_eval, "wb"))
    devBunch = build_features(config, config.dev_file, "dev", tokenizer)
    print("save dev bunch")
    pickle.dump(devBunch, open(config.dev_eval, "wb"))



