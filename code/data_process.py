#!/usr/bin/env python3
#coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'

import os
import sys
import re
import pandas as pd

def data_trans():
    fname = 'data/data_train.xlsx'
    df = pd.read_excel(fname)
    print(df.info())
    # 内容处理
    df.fillna('', inplace=True)

    # 对所有字符类字段进行过滤
    for col in df.columns:
        if df[col].dtype==object:
            df[col] = df[col].apply(lambda x:re.sub('([\n\t\r])', "", x))


    # 合并字段： age diseaseName conditionDesc title hopeHelp
    df['text'] = df['age'] + ';' + df['diseaseName'] + ';' + df['conditionDesc'] + ';' + df['title'] + ';' + df['hopeHelp']
    df_out = df[['text','label_i','label_j']]
    # 保存
    df_out.to_csv('data/all_data.csv', index=0)
    
    # 打乱，拆分
    df_out = df_out.sample(frac=1)
    length = df_out.shape[0]
    cut_train = int(length*0.8)
    df_train = df_out.head(cut_train)
    df_dev = df_out.tail(length-cut_train)

    # 保存
    df_train.to_csv('data/train.tsv', index=0, header=None, sep='\t')
    df_dev.to_csv('data/dev.tsv', index=0, header=None, sep='\t')
    print('训练数据已保存')


if __name__ == '__main__':
    pass
    data_trans()
