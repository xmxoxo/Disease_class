#!/usr/bin/env python3
#coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'

import argparse
import os
import sys
import re
import pandas as pd
import numpy as np

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
    ff = df_out[df_ou['label_j']==-1]
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
    ff = df_out[df_ou['label_j']==-1]
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
    # 读原始数据
    df = pd.read_csv(datname, sep='\t', header=None)
    df.columns = ['txt', 'olabel_i','olabel_j']
    df = df[['txt', 'olabel_i']]
    print(df.info())
    pl()

    # 读取预测结果
    df_pred = pd.read_csv(predfname)
    print(df_pred.info())
    pl()
    # 删除列
    df_pred.drop(['label_i', 'prob_i'], axis=1)
    
    # 连接两个表
    df_out = pd.concat([df, df_pred], axis=1)
    # 过滤预测置信度
    prob_low = df_out[df_out['prob_j'] < 0.5]
    df_out = df_out.drop(index=prob_low.index)

    # 提取最终结果列
    df_out = df_out[['txt', 'olabel_i','label_j']]
    df_out.columns = ['txt', 'label_i','label_j']

    # todo: 与原始训练集合并
    pass

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

def cut_data(df_out, scale=0.8):
    # 打乱，拆分
    df_out = df_out.sample(frac=1)
    length = df_out.shape[0]
    cut_train = int(length*scale)
    df_train = df_out.head(cut_train).copy()
    df_dev = df_out.tail(length-cut_train).copy()
    return df_train, df_dev

def data_trans_pipe(outpath):
    # 单任务独立模型数据处理
    mkfold(outpath)

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
    df['content'] = '年龄:' + df['age'] + ';' + df['diseaseName'] + ';' + df['conditionDesc'] + ';' + df['title'] + ';' + df['hopeHelp']
    df_out = df[['content','label_i','label_j']]
    # 保存所有数据
    df_out.to_csv(os.path.join(outpath, 'data_train.csv'), index=0)
    
    # 打乱，拆分, 保存
    mkfold(os.path.join(outpath, './label_i/'))
    df_out_i = df_out[['content','label_i']]
    df_train, df_dev = cut_data(df_out_i, scale=0.9)
    df_train.to_csv(os.path.join(outpath, 'label_i/train.tsv'), index=0, header=None, sep='\t')
    df_dev.to_csv(os.path.join(outpath, 'label_i/dev.tsv'), index=0, header=None, sep='\t')
    
    mkfold(os.path.join(outpath, './label_j/'))
    df_out_j = df_out[['content','label_j']]
    # 删除-1的样本
    ff = df_out_j[df_out_j['label_j']==-1]
    df_out_j = df_out_j.drop(index=ff.index)
    # 打乱，拆分, 保存
    df_train, df_dev = cut_data(df_out_j, scale=0.9)
    df_train.to_csv(os.path.join(outpath, 'label_j/train.tsv'), index=0, header=None, sep='\t')
    df_dev.to_csv(os.path.join(outpath, 'label_j/dev.tsv'), index=0, header=None, sep='\t')
    
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
    df['content'] = '年龄:' + df['age'] + ';' + df['diseaseName'] + ';' + df['conditionDesc'] + ';' + df['title'] + ';' + df['hopeHelp']
    df['label'] = [0] * df.shape[0]

    # 保存测试数据
    df_out = df[['content','label']]
    df_out.to_csv(os.path.join(outpath, 'label_i/test.tsv'), index=0, header=None, sep='\t')
    df_out.to_csv(os.path.join(outpath, 'label_j/test.tsv'), index=0, header=None, sep='\t')
    print('预测数据已保存')


def merge_pipe(fname):
    ''' 合并pipe预测结果
    '''

    dat_i = os.path.join(fname, 'label_i/submit.csv') #label_i_b
    dat_j = os.path.join(fname, 'label_j/submit.csv')

    df_i = pd.read_csv(dat_i)   #, sep='\t'
    df_j = pd.read_csv(dat_j)   #, sep='\t'

    df_out = pd.concat([df_i, df_j], axis=1)
    df_out.columns = ['id', 'label_i', 'id2', 'label_j']
    df_out = df_out[['id', 'label_i', 'label_j']]
    print(df_out.info())
    # 保存
    outfname = os.path.join(fname, 'submit.csv')
    df_out.to_csv(outfname,index=0)
    print('预测结果文件已合并保存至:%s' % outfname)


def text_process(df):
    ''' 数据预处理通用部分
    '''
    # 内容处理
    df.fillna(' ', inplace=True)

    # 对所有字符类字段进行过滤
    for col in df.columns:
        if df[col].dtype==object:
            df[col] = df[col].apply(lambda x:re.sub('([\n\t\r])', "", x))

    return df


def data_trans_split(outpath):
    ''' 串接模型数据处理
    '''
    # 
    mkfold(outpath)

    fname = 'data/data_train.xlsx'
    df = pd.read_excel(fname)
    print(df.info())
    # 基础数据处理
    df = text_process(df)
    #df_out = df.drop('id', axis=1)
    df_out = df[['age','diseaseName','title','conditionDesc','hopeHelp','label_i','label_j']]
    # 年龄字段处理
    df_out['age'] = '年龄:' + df_out['age']

    # 保存所有数据
    outf = os.path.join(outpath, 'data_train.csv')
    df_out.to_csv(outf, index=0)

    # 2022/6/30 把 label_j 列中为-1值的样本提取出来, 另存为'label_j.tsv'
    ff = df_out[df_out['label_j']==-1]
    ff.to_csv(os.path.join(outpath, 'label_j.tsv'), index=0, header=None, sep='\t')
    print('ff.shape:', ff.shape)
    pl()
    # 删除-1的样本
    df_out = df_out.drop(index=ff.index)
    print(df_out.info())

    # 打乱，拆分 保存
    df_train, df_dev = cut_data(df_out, scale=0.9)
    df_train.to_csv(os.path.join(outpath, 'train.tsv'), index=0, header=None, sep='\t')
    df_dev.to_csv(os.path.join(outpath, 'dev.tsv'), index=0, header=None, sep='\t')
    print('训练数据已保存')
    
    # -----------------------------------------
    # 处理预测数据
    fname = 'data/data_test.xlsx'
    df = pd.read_excel(fname)
    print(df.info())
    # 基础数据处理
    df = text_process(df)

    df['age'] = '年龄:' + df['age']
    df['label_i'] = [0] * df.shape[0]
    df['label_j'] = [0] * df.shape[0]

    # 保存测试数据
    df_out = df[['age','diseaseName','title','conditionDesc','hopeHelp','label_i','label_j']]
    df_out.to_csv(os.path.join(outpath, 'test.tsv'), index=0, header=None, sep='\t')
    print('预测数据已保存')


def analyze_data(df, column):
    import matplotlib.pyplot as plt
    # 计算平均值的MSE，用于表示数据的离散程度
    AVE_MSE = lambda x: np.log(np.sum(np.power(( np.ones_like(x)*np.average(x) - x),2))/len(x))
    
    ave_mse = AVE_MSE(df[column].values.tolist())
    print('ave_mse:%f' % ave_mse)
    return 
    data = df[column].value_counts()

    #print(data)
    #pl()
    #xy = data.values.tolist()
    #fig, ax = plt.subplots(figsize=(8,6))
    #plt.plot(xy)
    # 分布图
    plt.bar(data.index, data)
    plt.title(column)
    plt.ylabel('Count',fontsize=12)
    plt.xlabel('Value',fontsize=12)
    plt.show()

def data_analizy():
    fname = 'data/data_train.xlsx'
    df = pd.read_excel(fname)

    analyze_data(df, 'label_i')
    # 删除-1的样本
    ff = df[df['label_j']==-1]
    df_out = df.drop(index=ff.index)
    analyze_data(df_out, 'label_j')
    analyze_data(df, 'label_j')
       

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

    if task=='data_trans_pipe':
        data_trans_pipe(outpath)

    if task=='merge_pipe':
        merge_pipe(fname)

    if task=='data_trans_split':
        data_trans_split(outpath)

    if task=='data_analizy':
        data_analizy()
