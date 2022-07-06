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
from bert4keras.snippets import sequence_padding, DataGenerator
from keras.layers import Dropout, merge, Input, Lambda
from keras.layers import Lambda, Dense, Bidirectional, LSTM, SpatialDropout1D
from keras.models import Model,load_model

parser = argparse.ArgumentParser(description='单句子分类模型')
parser.add_argument('--task', type=str, default="train", help='train,eval,predict,online')
parser.add_argument('--data_path', type=str, default="./data/", help='data path')
parser.add_argument('--model_outpath', type=str, default="./output/", help='model outpath')
parser.add_argument('--bert_path', type=str, default='', help='bert_path')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size=32')
parser.add_argument('--epochs', type=int, default=10, help='epochs=10')
parser.add_argument('--lr', type=float, default=1e-5, help='learning_rate')
#parser.add_argument('--label_dict', type=str, default='', help='分类字典')
parser.add_argument('--pred_file', type=str, default='', help='预测文件')
parser.add_argument('--pred_outfile', type=str, default='', help='预测输出文件')
parser.add_argument('--debug', type=int, default=0, help='debug')
#parser.add_argument('--num_classes', type=int, default=0, help='num_classes')
parser.add_argument('--idx', type=int, default=0, help='idx')

args = parser.parse_args()
task = args.task
epochs = args.epochs
learning_rate = args.lr
batch_size = args.batch_size
data_path = args.data_path
bert_path = args.bert_path
#label_dict_file = args.label_dict
debug = args.debug
idx = args.idx
#num_classes = args.num_classes
num_classes_list = [20, 61]
num_classes = num_classes_list[idx]

if num_classes==0:
    print('num_classes:', num_classes)
    sys.exit()


maxlen = 256
set_gelu('tanh')  # 切换gelu版本

if bert_path == '':
    if os.name=='nt':
        bert_path = r'F:\models\chinese_L-12_H-768_A-12'
    else:
        bertbase = 'chinese_L-12_H-768_A-12'
        roberta = 'chinese_roberta_wwm_large_ext_L-24_H-1024_A-16'
        bert_path = '/mnt/sda1/models/' + bertbase

# 不加载模型时，自动创建输出目录
model_outpath = args.model_outpath

'''
# 分类字典默认为 数据目录下的 labels.txt
if label_dict_file == '':
    fname = os.path.join(data_path, 'labels.txt')
    if os.path.exists(fname):
        label_dict_file = fname
'''

# preload 
if task == 'train':
    preload = 0 
else:
    preload = 1

# 生成模型文件名
model_file_h5 = os.path.join(model_outpath, 'model.h5')
model_file_weight = os.path.join(model_outpath, 'model.weights')

# 预训练模型参数
config_path = os.path.join(bert_path, 'bert_config.json')
checkpoint_path = os.path.join(bert_path, 'bert_model.ckpt')
dict_path = os.path.join(bert_path, 'vocab.txt')

def load_catalog(filename):
    ''' 加载标签字典，返回列表
    '''
    D = []
    with open(filename, encoding='utf-8') as f:
        f = f.read()
        D = list(filter(None, f.split('\n')))
    return D


# 加载标签字典
# categories = load_catalog(label_dict_file)
# 自动计算分类数量 
#num_classes = len(categories)
print('num_classes:', num_classes)


# 保存文本信息到文件
def savetofile(txt, filename, encoding='utf-8'):
    try:
        with open(filename, 'w', encoding=encoding) as f:  
            f.write(str(txt))
        return 1
    except :
        return 0

# 自动生成目录
def mkfold(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

def load_data(filename):
    """加载数据
    单条格式：(文本+TAB+标签id)
    输出格式：(文本, 标签id)
    """
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.read().split('\n')
        for l in data:
            x = l.strip().split('\t')
            if len(x)==1:
                text, label = x[0], 0 
            if len(x)>=2:
                text, label = x[:2]
                if not label.isnumeric(): continue
            if text:
                # 内容截取
                D.append((text, int(label)))
    return D

def load_all_data(path):
    print('正在加载数据集...')
    train_data = load_data(os.path.join(path, 'train.tsv'))
    valid_data = load_data(os.path.join(path, 'dev.tsv'))
    test_data = load_data(os.path.join(path, 'test.tsv'))
   
    return train_data,valid_data,test_data

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

def txt2sample(text):
    ''' 单文本编码
    '''
    batch_token_ids, batch_segment_ids, batch_labels = [], [], []
    token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)

    batch_token_ids.append(token_ids)
    batch_segment_ids.append(segment_ids)

    batch_token_ids = sequence_padding(batch_token_ids)
    batch_segment_ids = sequence_padding(batch_segment_ids)
    return [batch_token_ids, batch_segment_ids]#, batch_labels

# 创建基础模型
def BaseModel():
    '''搭建编码层网络,用于权重共享'''
    bert = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        return_keras_model=False,
    )
    # 输出层使用mean
    #output = Lambda(lambda x: K.mean(x, axis=1), name='MEAN-token')(bert.model.output)
    output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)

    # 降维
    # output = Dense(units=256)(output)
    return Model(bert.model.input, output, name="BERT_base")

# 创建组合
def CModel(num_classes):
    model_base = BaseModel()

    # 输入和输出
    X1 = Input(shape=(None,), name='Input-Token')
    X2 = Input(shape=(None,), name='Input-Segment')

    X = model_base([X1, X2])
    output = Dense(units=num_classes, activation='softmax')(X)

    model = Model([X1, X2], output, name="ComboModel")
    return model

print('正在创建模型...')
model = CModel(num_classes)
model.summary()
#sys.exit()

# 派生为带分段线性学习率的优化器。
# 其中name参数可选，但最好填入，以区分不同的派生优化器。
AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')

model.compile(
    loss='sparse_categorical_crossentropy', 
    #sparse_categorical_crossentropy binary_crossentropy categorical_crossentropy
    #optimizer=Adam(learning_rate),  # 用足够小的学习率
    optimizer=AdamLR(learning_rate=learning_rate, lr_schedule={
        1000: 1,
        2000: 0.1
    }),
    metrics=['accuracy'],
)

def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true)
        y_pred = y_pred.argmax(axis=1)
        y_true = y_true[:, 0]  # 取第1列, 其实是把[[1],[0],[3]]的结果转成 [1,0,3]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    if total == 0:
        ret = 0
    else:
        ret = right / total
    
    return ret

class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            # 创建模型输出目录
            mkfold(model_outpath)
            # 保存模型
            model.save_weights(model_file_weight)

        #test_acc = evaluate(test_generator)
        print(u'val_acc: %.5f, best_val_acc: %.5f\n' %(val_acc, self.best_val_acc))

# 预测测试集并保存
def predict_test(model, datfile, outfile):
    # 加载数据
    pred_data = load_data(datfile)
    print('pred_data:', len(pred_data))

    pred_generator = data_generator(pred_data, batch_size)
    
    # 批量预测
    pred = model.predict_generator(pred_generator.forfit(random=False), steps=len(pred_generator), verbose=1)
    y_pred = np.argmax(pred, axis=1)
    # 预测概率
    ll = np.arange(y_pred.shape[0]) 
    y_prob = pred[ll, y_pred]

    # 保存
    df_pred = pd.DataFrame({"id": range(len(y_pred)), "label":y_pred}, "prob":y_prob)
    df_pred.to_csv(outfile, index=0)


# 通用训练曲线生成 
def plot_loss (model_path):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    history_file = os.path.join(model_path, 'model_fit.log')
    with open(history_file,'r', encoding='utf-8') as f:  
        data = f.read()
    history = eval(data)
    keys = list(history.keys())

    for key in history.keys():
        fig = plt.figure(figsize=(10,5))
        plt.plot(history[key]) #, 'r-'
      
        plt.title('Model Report')
        plt.ylabel('value')
        plt.xlabel('epoch')
        plt.legend([key], loc='center right')
        plt.savefig(os.path.join(model_path, 'result_%s.png' % key))
        plt.close()

    print('训练曲线图已保存。')

def model_score(y_true, y_pred, idx=0):
    from sklearn.metrics import f1_score, accuracy_score, recall_score
    
    acc = accuracy_score(y_true, y_pred)
    recall = accuracy_score(y_true, y_pred)
    if idx==0:
        f1 = f1_score(y_true, y_pred, average='macro')  #weighted  macro micro)
        print('Accuracy:%.4f Recall:%.4f F1-macro:%.4f' % (acc, recall, f1))
    else:
        f1 = f1_score(y_true, y_pred, average='micro')  #weighted  macro micro)
        print('Accuracy:%.4f Recall:%.4f F1-micro:%.4f' % (acc, recall, f1))
    
    return f1

def calc_test_acc(y_pred, test_y):
    '''测试集准确率, 输出分类报告及混淆矩阵
    '''
    from sklearn.metrics import classification_report, confusion_matrix
    repname = os.path.join(model_outpath, 'report.txt')
    repo = []

    if type(y_pred)!=np.ndarray:
        y_pred = np.array(y_pred)
    if type(test_y)!=np.ndarray:
        test_y = np.array(test_y)

    total = sum(y_pred==test_y)
    test_acc = total/test_y.shape[0]

    repo.append('test acc: %f'%( test_acc))
    # 生成报告 
    class_report_cat = classification_report(test_y, y_pred, digits=3)
    repo.append('-'*40)
    repo.append(str(class_report_cat))
    # 生成混淆矩阵 
    matrix = confusion_matrix(test_y, y_pred)
    repo.append('混淆矩阵'.center(40,'-'))
    repo.append( str(matrix.tolist()))
    repo_txt = '\n'.join(repo)
    savetofile(repo_txt, repname)

    print('正在生成混淆矩阵图...')
    import seaborn as sns
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    target_names = [str(i) for i in range(num_classes)]
    fig, ax = plt.subplots(figsize=(14,12))
    sns.heatmap(matrix, annot=True, fmt='d',
                xticklabels=list(range(num_classes)), yticklabels=target_names )
    plt.ylabel('Sample',fontsize=12)
    plt.xlabel('Predict',fontsize=12)

    picname = os.path.join(model_outpath, 'model_matrix.png')
    plt.savefig(picname)
    plt.close()
    print('混淆矩阵图已保存')

    #return y_pred, test_acc
 

if __name__ == '__main__':
    
    def save_run():
        
        # 保存运行参数
        logfile = 'run.log'
        logtxt = []
        logtxt.append ('训练命令:%s'% task)
        logtxt.append ('模型输出目录:%s'% model_outpath)
        logtxt.append ('训练数据目录:%s'% data_path)
        logtxt.append ('预训练模型目录:%s'%bert_path)
        #logtxt.append ('预加载模型:%s, %s'% (preload, model_path))
        logtxt.append ('训练轮数:%d'% epochs)
        logtxt.append ('学习率:%f'% learning_rate)
        logtxt.append ('batch_size:%d'% batch_size)
        
        savetofile('\n'.join(logtxt),  os.path.join(model_outpath, logfile) ) 

    if task == 'train':
        save_run()
        if preload==1:
            model.load_weights(model_file_weight)
            print('模型已加载。')

        evaluator = Evaluator()

        # 加载数据集
        train_data, valid_data, test_data =load_all_data(data_path)

        # 转换数据集
        train_generator = data_generator(train_data, batch_size)
        valid_generator = data_generator(valid_data, batch_size)
        test_generator = data_generator(test_data, batch_size)


        # 训练模型
        history_fit = model.fit(
            train_generator.forfit(),
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            callbacks=[evaluator]
        )
        print('正在保存训练数据...')
        with open(os.path.join(model_outpath, 'model_fit.log'), 'w') as f:
            f.write(str(history_fit.history))

        # 画训练曲线图
        plot_loss(model_outpath)

    if task in ['train', 'predict']: 
        # 批量预测
        print('正在预测数据...')
        
        # 加载模型 
        model.load_weights(model_file_weight)

        # 加载待预测数据
        pred_file = args.pred_file
        pred_outfile = args.pred_outfile

        # 默认的数据
        if pred_file == '':
            pred_file = os.path.join(data_path, 'test.tsv')
        if pred_outfile == '':
            pred_outfile = os.path.join(model_outpath, 'submit.csv')
        
        predict_test(model, pred_file, pred_outfile)
        print('预测结果已保存:%s' % pred_outfile)
    
    if task in ['train', 'eval']:
        # 加载数据
        if task == 'eval':
            train_data, valid_data, test_data =load_all_data(data_path)

        # 加载验证集
        test_generator = data_generator(valid_data, batch_size)

        # 加载模型
        print('正在加载模型...')
        model.load_weights(model_file_weight)
        #if not os.path.exists(model_file_h5):
        #    model.save(model_file_h5)

        print('正在验证数据集...')
        begin = time.time()
        y_true = [b for a,b in valid_data]
        
        pred = model.predict_generator(test_generator.forfit(random=False), 
                        steps=len(test_generator), verbose=1)

        y_pred = np.argmax(pred, axis=1)
        total_time = round((time.time() - begin)*1000,3)
        print('预测数据用时%.3f毫秒.'%total_time)

        #calc_test_acc(y_pred, test_y)
        model_score(y_true, y_pred, idx)

    if task == 'online': # 实时预测        
        '''
        # 加载label字典
        label_dict = {}
        if label_dict_file:
            label_dict = load_catalog(label_dict_file)
        
        '''
        # 加载模型
        print('正在加载模型...')
        model.load_weights(model_file_weight)
        if not os.path.exists(model_file_h5):
            model.save(model_file_h5)

        # 交互式预测: 
        while 1:
            try:
                text = input('请输入文本[Q退出]:').strip()
                if not text: continue
                if text in ['q', 'Q', 'quit', 'Quit'] : break
                x = txt2sample(text)
                #print('sample:\n',x)
                pred = model.predict(x)
                y_pred = pred.argmax(axis=1)[0]
                #print('pred:', pred)
                #print('y_pred:', y_pred)
                if label_dict:
                    result = label_dict[y_pred]
                else:
                    result = str(y_pred)
                print('预测结果:%s(%s)' % (result, y_pred))
                print('-'*40)
            except Exception as e:
                print('运行错误:', e)

