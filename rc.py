#! -*- coding: utf-8 -*-
# 百度LIC2021的机器阅读理解赛道，非官方baseline
# 直接用RoFormer+Softmax预测首尾
# BASE模型在第一期测试集上F1=53.456和EM=46.523，略低于官方baseline

import json, os
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import truncate_sequences, open, to_array
from keras.layers import Layer, Dense, Permute
from keras.models import Model
from tqdm import tqdm
import jieba
jieba.initialize()

# 基本信息
maxlen = 256
epochs = 20
batch_size = 32
learing_rate = 2e-5

# bert配置
config_path = '/root/kg/bert/chinese_roformer_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/root/kg/bert/chinese_roformer_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/root/kg/bert/chinese_roformer_L-12_H-768_A-12/vocab.txt'


def truncate_strings(*strings):
    """截断字符串至总长度不超过maxlen
    """
    strings = [list(s) for s in strings]
    truncate_sequences(maxlen, -1, *strings)
    return [''.join(s) for s in strings]


def load_data(filename):
    """读取数据
    格式：(问题, 左篇章, 答案, 右篇章)
    """
    D = []
    data = json.load(open(filename))
    data = [i for d in data['data'] for i in d['paragraphs']]
    for d in data:
        c = d['context']
        for qa in d['qas']:
            q = qa['question']
            for a in qa['answers']:
                if a['answer_start'] == -1:
                    D.append(truncate_strings(q, '', '', c))
                else:
                    start = a['answer_start']
                    end = start + len(a['text'])
                    D.append(
                        truncate_strings(q, c[:start], c[start:end], c[end:])
                    )
    return D


# 读取数据
train_data = load_data('../datasets/train.json')
valid_data = load_data('../datasets/dev.json')

# 建立分词器
tokenizer = Tokenizer(
    dict_path,
    do_lower_case=True,
    pre_tokenize=lambda s: jieba.lcut(s, HMM=False)
)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, item in self.sample(random):
            token_ids = tokenizer.encode(item[0])[0]
            token_ids += tokenizer.encode(item[1])[0][1:-1]
            if item[2]:
                a_ids = tokenizer.encode(item[2])[0][1:-1]
                start, end = len(token_ids), len(token_ids) + len(a_ids) - 1
                token_ids += a_ids
            else:
                start, end = 0, 0
            token_ids += tokenizer.encode(item[3])[0][1:-1]
            labels = [[start], [end]]
            batch_token_ids.append(token_ids)
            batch_segment_ids.append([0] * len(token_ids))
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


class MaskedSoftmax(Layer):
    """在序列长度那一维进行softmax，并mask掉padding部分
    """
    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            mask = K.expand_dims(mask, 2)
            inputs = inputs - (1.0 - mask) * 1e12
        return K.softmax(inputs, 1)


model = build_transformer_model(config_path, checkpoint_path, model='roformer')

output = Dense(2)(model.output)
output = MaskedSoftmax()(output)
output = Permute((2, 1))(output)

model = Model(model.input, output)
model.summary()


def sparse_categorical_crossentropy(y_true, y_pred):
    # y_true需要重新明确一下shape和dtype
    y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
    y_true = K.cast(y_true, 'int32')
    y_true = K.one_hot(y_true, K.shape(y_pred)[2])
    # 计算交叉熵
    return K.mean(K.categorical_crossentropy(y_true, y_pred))


def sparse_accuracy(y_true, y_pred):
    # y_true需要重新明确一下shape和dtype
    y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
    y_true = K.cast(y_true, 'int32')
    # 计算准确率
    y_pred = K.cast(K.argmax(y_pred, axis=2), 'int32')
    return K.mean(K.cast(K.equal(y_true, y_pred), K.floatx()))


model.compile(
    loss=sparse_categorical_crossentropy,
    optimizer=Adam(learing_rate),
    metrics=[sparse_accuracy]
)


def extract_answer(question, context):
    """抽取答案函数
    """
    question, context = truncate_strings(question, context)
    q_token_ids = tokenizer.encode(question)[0]
    c_token_ids = tokenizer.encode(context)[0]
    token_ids = q_token_ids + c_token_ids[1:]
    segment_ids = [0] * len(token_ids)
    c_tokens = tokenizer.tokenize(context)[1:-1]
    mapping = tokenizer.rematch(context, c_tokens)
    token_ids, segment_ids = to_array([token_ids], [segment_ids])
    probas = model.predict([token_ids, segment_ids])[0]
    probas[:, 1:len(q_token_ids)] *= 0
    probas[:, -1] *= 0
    start_end, score = None, -1
    for start, p_start in enumerate(probas[0]):
        for end, p_end in enumerate(probas[1]):
            if end >= start:
                if p_start * p_end > score:
                    start_end = (start, end)
                    score = p_start * p_end
    start, end = start_end
    if start == 0:
        return ''
    else:
        start, end = start - len(q_token_ids), end - len(q_token_ids)
        return context[mapping[start][0]:mapping[end][-1] + 1]


def predict_to_file(in_file, out_file):
    """预测结果到文件，方便提交
    """
    answers = {}
    data = json.load(open(in_file))
    data = [i for d in data['data'] for i in d['paragraphs']]
    for d in tqdm(data):
        c = d['context']
        for qa in d['qas']:
            q = qa['question']
            a = extract_answer(q, c) or 'no answer'
            answers[qa['id']] = a
    json.dump(
        answers,
        open(out_file, 'w', encoding='utf-8'),
        indent=4,
        ensure_ascii=False
    )


def evaluate(filename):
    """评测函数（官方提供评测脚本evaluate.py）
    """
    predict_to_file(filename, filename + '.pred.json')
    metrics = json.loads(
        os.popen(
            'python ../datasets/evaluate.py %s %s' %
            (filename, filename + '.pred.json')
        ).read().strip()
    )
    metrics['F1'] = float(metrics['F1'])
    metrics['EM'] = float(metrics['EM'])
    return metrics


class Evaluator(keras.callbacks.Callback):
    """评估和保存模型
    """
    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, epoch, logs=None):
        metrics = evaluate('../datasets/dev.json')
        if metrics['F1'] >= self.best_val_f1:
            self.best_val_f1 = metrics['F1']
            model.save_weights('best_model.weights')
        metrics['BEST F1'] = self.best_val_f1
        print(metrics, '\n')


if __name__ == '__main__':

    train_generator = data_generator(train_data, batch_size)
    evaluator = Evaluator()

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )

else:

    model.load_weights('best_model.weights')
    # predict_to_file('../datasets/test1.json', 'rc_pred.json')
