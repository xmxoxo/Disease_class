#!/usr/bin/env python3
#coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'

import argparse
import os
import sys
import re
import pandas as pd

pl = lambda x='-': print(x*40)

# 自动生成目录
def mkfold(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

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
    
    '''
    # label_j字段值+1；由-1~60变为 0~61
    df['label_j'] = df['label_j'] +1 
    '''

    # 合并字段： age diseaseName conditionDesc title hopeHelp
    df['content'] = df['age'] + ';' + df['diseaseName'] + ';' + df['conditionDesc'] + ';' + df['title'] + ';' + df['hopeHelp']
    df_out = df[['content','label_i','label_j']]
    # 保存所有数据
    df_out.to_csv('data/data_train.csv', index=0)

    # 2022/6/30 把 label_j 列中为-1值的样本提取出来, 另存为'label_j.tsv'
    ff = df_out[df['label_j']==-1]
    ff.to_csv('data/label_j.tsv', index=0, header=None, sep='\t')
    print('ff.shape:', ff.shape)
    pl()
    # 删除-1的样本
    df_out = df_out.drop(index=ff.index)
    print(df_out.info())

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

def data_trans_age():
    ''' 带年龄的数据预处理
    '''
    mkfold('./data_age')
    fname = 'data/data_train.xlsx'
    df = pd.read_excel(fname)
    print(df.info())
    # 内容处理
    df.fillna('', inplace=True)

    # 对所有字符类字段进行过滤
    for col in df.columns:
        if df[col].dtype==object:
            df[col] = df[col].apply(lambda x:re.sub('([\n\t\r])', "", x))
    
    # 年龄字段处理
    df['age_value'] = df['age'].apply(lambda x: int(x.replace('+', '')))

    # 合并字段： diseaseName conditionDesc title hopeHelp
    df['content'] = df['diseaseName'] + ';' + df['conditionDesc'] + ';' + df['title'] + ';' + df['hopeHelp']
    df_out = df[['age_value', 'content','label_i','label_j']]
    # 保存所有数据
    df_out.to_csv('data_age/data_train.csv', index=0)

    # 2022/6/30 把 label_j 列中为-1值的样本提取出来, 另存为'label_j.tsv'
    ff = df_out[df['label_j']==-1]
    ff.to_csv('data_age/label_j.tsv', index=0, header=None, sep='\t')
    print('ff.shape:', ff.shape)
    pl()
    # 删除-1的样本
    df_out = df_out.drop(index=ff.index)
    print(df_out.info())

    # 打乱，拆分
    df_out = df_out.sample(frac=1)
    length = df_out.shape[0]
    cut_train = int(length*0.8)
    df_train = df_out.head(cut_train)
    df_dev = df_out.tail(length-cut_train)

    # 保存
    df_train.to_csv('data_age/train.tsv', index=0, header=None, sep='\t')
    df_dev.to_csv('data_age/dev.tsv', index=0, header=None, sep='\t')
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

    # 年龄字段处理
    df['age_value'] = df['age'].apply(lambda x: int(x.replace('+', '')))
    
    # 合并字段： age diseaseName conditionDesc title hopeHelp
    df['content'] = df['diseaseName'] + ';' + df['conditionDesc'] + ';' + df['title'] + ';' + df['hopeHelp']
    df['label_i'] = [0] * df.shape[0]
    df['label_j'] = [0] * df.shape[0]

    df_out = df[['age_value', 'content', 'label_i', 'label_j']]
    # 保存测试数据
    df_out.to_csv('data_age/test.tsv', index=0, header=None, sep='\t')
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

def merge_predict(fname, outpath):
    '''合并-1标签数据与预测结果,并生成训练数据
    '''
    from shutil import copyfile
    mkfold(outpath)

    datname, predfname = fname.split(',')
    df = pd.read_csv(datname, sep='\t', header=None)
    df.columns = ['txt', 'olabel_i','olabel_j']
    print(df.info())
    pl()

    df_pred = pd.read_csv(predfname)
    print(df_pred.info())
    pl()
    
    df_out = pd.concat([df, df_pred], axis=1)
    df_out = df_out[['txt', 'olabel_i','label_j']]
    df_out.columns = ['txt', 'label_i','label_j']
    print(df_out.info())
    pl()
    print(df_out.head())
    pl()

    # 保存结果
    outfile = os.path.join(outpath, 'train.tsv')
    df_out.to_csv(outfile, index=0, header=None, sep='\t')
    print('合并结果已保存到:%s'%outfile)
    
    # 把原始的测试集和验证集复制过来
    p, f = os.path.split(datname)
    flist = ['dev.tsv','test.tsv']
    for fn in flist:
        sf = os.path.join(p, fn)
        tf = os.path.join(outpath, fn)
        copyfile(sf, tf)                

    print('训练数据已生成。')

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
    if task=='merge_predict':
        merge_predict(fname, outpath)

    if task=='data_trans_age': # 年龄字段独立提取特征
        data_trans_age()
