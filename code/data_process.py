#!/usr/bin/env python3
#coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'

import argparse
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
    # label_j字段值+1；由-1~60变为 0~61
    df['label_j'] = df['label_j'] +1 

    # 合并字段： age diseaseName conditionDesc title hopeHelp
    df['content'] = df['age'] + ';' + df['diseaseName'] + ';' + df['conditionDesc'] + ';' + df['title'] + ';' + df['hopeHelp']
    df_out = df[['content','label_i','label_j']]
    # 保存
    df_out.to_csv('data/data_train.csv', index=0)
    
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
    
    # -----------------------------------------
    # 处理预测数据
    fname = 'data/data_test.xlsx'
    df = pd.read_excel(fname)
    print(df.info())
    # 内容处理
    df.fillna('', inplace=True)

    # 对所有字符类字段进行过滤
    for col in df.columns:
        if df[col].dtype==object:
            df[col] = df[col].apply(lambda x:re.sub('([\n\t\r])', "", x))
    # 合并字段： age diseaseName conditionDesc title hopeHelp
    df['content'] = df['age'] + ';' + df['diseaseName'] + ';' + df['conditionDesc'] + ';' + df['title'] + ';' + df['hopeHelp']
    df['label_i'] = [0] * df.shape[0]
    df['label_j'] = [0] * df.shape[0]

    df_out = df[['content','label_i','label_j']]
    # 保存测试数据
    df_out.to_csv('data/test.tsv', index=0, header=None, sep='\t')
    print('预测数据已保存')


def submit_test():
    # 提交数据分析
    fname = 'models/submit.csv'
    df = pd.read_csv(fname)
    print(df.info())
    dd = df['label_j'].value_counts()
    print(dd.index)
    print('-'*40)
    print(list(zip(dd.index, dd.tolist())))
    print('-'*40)
    print(sorted(dd.index))

if __name__ == '__main__':
    pass
    parser = argparse.ArgumentParser(description='数据预处理')
    parser.add_argument('--task', type=str, required=True, default="", help='处理命令')
    parser.add_argument('--fname', type=str, default="", help='处理文件或者目录')
    parser.add_argument('--outpath', type=str, default="", help='输出文件或目录')
    parser.add_argument('--topn', type=int, default=1000, help='topn')

    args = parser.parse_args()
    task = args.task
    fname = args.fname
    outpath = args.outpath
    topn = args.topn

    if task=='data_trans':
        data_trans()

    if task=='submit_test':
        submit_test()

