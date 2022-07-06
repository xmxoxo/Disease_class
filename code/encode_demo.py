#!/usr/bin/env python3
#coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'

import argparse
import os
import sys
import re
import time

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from bert4keras.backend import keras, set_gelu, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, DataGenerator, to_array
from keras.layers import Dropout, merge, Input, Lambda
from keras.layers import Lambda, Dense, Bidirectional, LSTM, Concatenate
from keras.utils.np_utils import to_categorical
from keras.models import Model,load_model


maxlen = 256

bert_path = r'F:\models\chinese_L-12_H-768_A-12'


# 预训练模型参数
config_path = os.path.join(bert_path, 'bert_config.json')
checkpoint_path = os.path.join(bert_path, 'bert_model.ckpt')
dict_path = os.path.join(bert_path, 'vocab.txt')

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


def encode(tokenizer, txt_list, maxlen):
    ''' 多文本编码器
    '''
    batch_token_ids, batch_segment_ids  = [], []
    for text in txt_list:
        token_ids, segment_ids = tokenizer.encode(text)
        if batch_token_ids != []:
            token_ids = token_ids[1:]
            segment_ids = segment_ids[1:]

        batch_token_ids.extend(token_ids)
        batch_segment_ids.extend(segment_ids)

        #batch_token_ids.append(token_ids)
        #batch_segment_ids.append(segment_ids)
    
    if len(batch_token_ids) > maxlen:
        pass
    return batch_token_ids, batch_segment_ids


txt_list = ['年龄:10+;','发高烧;','头痛，轻微的流鼻涕，发高烧，不咳嗽，喉咙不疼;','请问是什么病？;','请问是什么病？']


token_ids, segment_ids = encode(tokenizer, txt_list, maxlen)

print('token_ids:', len(token_ids), token_ids)
print('segment_ids:', len(segment_ids), segment_ids)

print('-'*40)
text = ''.join(txt_list)
token_ids, segment_ids = tokenizer.encode(text, maxlen=25)
print('token_ids:', len(token_ids), token_ids)
print('segment_ids:', len(segment_ids), segment_ids)


if __name__ == '__main__':
    pass


