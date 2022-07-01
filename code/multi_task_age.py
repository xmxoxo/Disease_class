#!/usr/bin/env python3
#coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'

'''
年龄+文本 融合模型
'''


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

parser = argparse.ArgumentParser(description='单文本多任务分类模型')
parser.add_argument('--task', type=str, default="train", help='train,eval,predict,online')
parser.add_argument('--data_path', type=str, default="./data_age/", help='data path')
parser.add_argument('--model_outpath', type=str, default="./model_age", help='model outpath')
parser.add_argument('--bert_path', type=str, default='', help='bert_path')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size=32')
parser.add_argument('--epochs', type=int, default=10, help='epochs=10')
parser.add_argument('--lr', type=float, default=1e-5, help='learning_rate')
parser.add_argument('--pred_file', type=str, default='data_age/test.tsv', help='预测文件')
parser.add_argument('--pred_outfile', type=str, default='./model_age/submit.csv', help='预测输出文件')
parser.add_argument('--preload_model', type=str, default='', help='预加载模型文件')
parser.add_argument('--debug', type=int, default=0, help='debug')
parser.add_argument('--frozen', type=int, default=-1, help='frozen')

args = parser.parse_args()
task = args.task
epochs = args.epochs
learning_rate = args.lr
batch_size = args.batch_size
data_path = args.data_path
model_outpath = args.model_outpath
bert_path = args.bert_path
#label_dict_file = args.label_dict
label_dict_file = ''
debug = args.debug
frozen = args.frozen
preload_model = args.preload_model

maxlen = 256
set_gelu('tanh')  # 切换gelu版本

if bert_path == '':
    if os.name=='nt':
        bert_path = r'F:\models\chinese_L-12_H-768_A-12'
    else:
        roberta = 'chinese_roberta_wwm_large_ext_L-24_H-1024_A-16'
        bertbase = 'chinese_L-12_H-768_A-12'
        bert_path = '/mnt/sda1/models/' + bertbase

if preload_model == '' :
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

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)
# -----------------------------------------

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
    单条格式：(年龄 文本   label_i  label_j)
    输出格式：(年龄, 文本, [label_i, label_j])
    """
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.read().split('\n')
        for l in data:
            x = l.strip().split('\t')
            if len(x)==2:
                age, text, label1, label2 = x[0], x[1], 0, 0
            if len(x)>=4:
                age, text, label1, label2 = x[:4]
            if text:
                age = int(age)
                label = [int(label1), int(label2)]
                D.append((age, text, label))
    return D#[:100]


def load_all_data(path):
    print('正在加载数据集...')
    train_data = load_data(os.path.join(path, 'train.tsv'))
    valid_data = load_data(os.path.join(path, 'dev.tsv'))
    test_data = load_data(os.path.join(path, 'test.tsv'))
   
    return train_data,valid_data,test_data


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_ages, batch_token_ids, batch_segment_ids, batch_labels = [], [], [], []
        for is_end, (age, text, label) in self.sample(random):
            batch_ages.append(age)
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(label)
            
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                
                batch_labels = [batch_labels[:,0], batch_labels[:,1]]
                yield batch_ages, [batch_token_ids, batch_segment_ids], batch_labels 
                batch_ages, batch_token_ids, batch_segment_ids, batch_labels = [], [], [], []

# 创建基础模型
def BaseModel(config_path, checkpoint_path):
    '''搭建编码层网络,用于权重共享'''
    bert = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        return_keras_model=False,
    )
    # 输出层使用mean
    #output = Lambda(lambda x: K.mean(x, axis=1), name='MEAN-token')(bert.model.output)
    output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)
    return Model(bert.model.input, output, name="BERT_base")

# 多任务模型
def MModel(config_path, checkpoint_path, num_classes):
    # BERT输入和输出
    X1 = Input(shape=(None,), name='Input-Token')
    X2 = Input(shape=(None,), name='Input-Segment')
    
    model_base = BaseModel(config_path, checkpoint_path)
    output = model_base([X1, X2])
    
    # 降维
    output = Dense(units=256, name='Layer_dense')(output)

    # 年龄特征
    age = Input(shape=(1,), name='AGE-Token')
    # 文本特征与年龄特征融合
    output = Concatenate(axis=1)([age, output])
    # print('output:', output.shape) #257 或者769

    out = []
    for i,num in enumerate(num_classes):
        tmp = Dense(units=num, activation='softmax', name='out_%d'%i)(output)
        out.append(tmp)

    model = Model([age, X1, X2], out, name="MModel")
    return model

print('正在创建模型...')
# 分类大小
num_classes = [20, 61]
model = MModel(config_path, checkpoint_path, num_classes)
model.summary()
print('model input:', model.input)
print('model output:', model.output)

'''
指定冻结层的索引号 -1 表示不冻结, 0,1表示冻
'''
if frozen >= 0:
    layer_name = ['out_0', 'out_1'][frozen] 
    # 冻住指定层 禁止训练
    layer_out_1 = model.get_layer(layer_name)
    layer_out_1.trainable = False

#sys.exit()

# 派生为带分段线性学习率的优化器。
# 其中name参数可选，但最好填入，以区分不同的派生优化器。
AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')

model.compile(
    #loss='binary_crossentropy', 
    
    #loss='sparse_categorical_crossentropy', 
    loss={'out_0': 'sparse_categorical_crossentropy','out_1': 'sparse_categorical_crossentropy'},
    loss_weights={'out_0':1, 'out_1': 1},
    
    #sparse_categorical_crossentropy binary_crossentropy categorical_crossentropy
    #optimizer=Adam(learning_rate),  # 用足够小的学习率
    optimizer=AdamLR(learning_rate=learning_rate, lr_schedule={
        1000: 1,
        2000: 0.1
    }),
    metrics=['acc'],
)

# 加载数据集
train_data, valid_data, test_data = load_all_data(data_path)

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)

'''
print('batch_size:', batch_size)
for txt, label in train_generator.forfit():
    print('label.shape:', label.shape)
    print(label)
    break;
sys.exit()
'''

def evaluate(data):
    '''
    total, right = 0., 0.
    y_trues = []
    y_preds = []
    for x_true, y_true in t_generator:
        y_pred = model.predict(x_true)
        #print('y_pred=', y_pred)

        y_pred = np.array([y_pred[0].argmax(axis=1), y_pred[1].argmax(axis=1)])
        y_preds.append(y_pred)

        y_pred = np.array(list(zip(*y_pred)))
        #y_pred = list(zip(*y_pred))
        #print('y_pred=', y_pred)
        #-----------------------------------------
        #print('y_true=', y_true)
        y_true = np.array((y_true))
        y_trues.append(y_true)
        #print('y_pred=', y_pred)
        y_true = np.array(list(zip(*y_true)))
        #print('y_true=', y_true)
        total += len(y_true)
        right += ((y_true == y_pred).sum(axis=1)==2).sum()
    if total == 0:
        ret = 0
    else:
        ret = right / total
    '''

    y_true = np.array([b for a,b in data])

    t_generator = data_generator(data, batch_size)
    pred = model.predict_generator(t_generator.forfit(random=False), 
                    steps=len(t_generator), verbose=0)

    y_pred = np.array([pred[0].argmax(axis=1), pred[1].argmax(axis=1)])
    y_pred = y_pred.T

    #print('y_true:', y_true.shape)
    #print('y_pred:', y_pred.shape)
    f1 = multitask_model_score(y_true, y_pred, show=0)
    return f1

class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_score = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_score = evaluate(valid_data) # valid_data valid_generator
        if val_score > self.best_val_score:
            self.best_val_score = val_score
            # 创建模型输出目录
            mkfold(model_outpath)
            # 保存模型
            model.save_weights(model_file_weight)

        #test_acc = evaluate(test_generator)
        print(u'val_F1: %.5f, best_val_F1: %.5f\n' %(val_score, self.best_val_score))

# 预测测试集并保存
def predict_test(model, pred_file, outfile):

    # 自动生成输出文件名
    if outfile == "":
        fname = 'submit_%s.tsv' % time.strftime('%Y%m%d_%H%M%S', time.localtime())
        p, f = os.path.split(pred_file)
        outfile = os.path.join(p, fname)

    # 加载数据
    pred_data = load_data(pred_file)
    print('pred_data:', len(pred_data))
    pred_generator = data_generator(pred_data, batch_size)
    
    # 批量预测
    pred = model.predict_generator(pred_generator.forfit(random=False), steps=len(pred_generator), verbose=1)

    y_pred = [pred[0].argmax(axis=1), pred[1].argmax(axis=1)]
    # 保存预测的提交结果
    df = pd.DataFrame({"id": range(len(y_pred[0])), "label_i":y_pred[0], "label_j":y_pred[1]})
    df.to_csv(outfile, index=0)
    print('提交文件已生成：%s'%outfile)


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
 
def multitask_model_score(y_true, y_pred, show=1):
    ''' 
    多任务模型的模型评价 用F1之和作为指标
    '''
    from sklearn.metrics import f1_score, accuracy_score, recall_score
    
    # 多少个任务
    task_count = y_true.shape[1]
    f1_final = 0
    scores = []
    for i in range(task_count):
        if show: print( ('Task: %d'%i).center(40, '-'))
        y_t = y_true[:, i]
        y_p = y_pred[:, i]
        acc = accuracy_score(y_t, y_p)
        recall = accuracy_score(y_t, y_p)
        if i==0:
            f1 = f1_score(y_t, y_p, average='macro')  #weighted  macro micro)
            if show: print('Accuracy:%.2f Recall:%.2f F1-macro:%.2f' % (acc, recall, f1))
        else:
            f1 = f1_score(y_t, y_p, average='micro')  #weighted  macro micro)
            if show: print('Accuracy:%.2f Recall:%.2f F1-micro:%.2f' % (acc, recall, f1))
                    
        f1_final += f1
        # 记录
        scores.append((acc, recall, f1)) 

    if show: 
        print('F1_total:%.4f\n' % f1_final)
    else:
        # 单行输出
        pass
        print('model score: %s F1_final:%.4f'% (scores, f1_final))

    return f1_final
    
if __name__ == '__main__':
    
    def save_run():
        
        # 保存运行参数
        logfile = 'run.log'
        logtxt = []
        logtxt.append ('训练命令:%s'% task)
        logtxt.append ('模型输出目录:%s'% model_outpath)
        logtxt.append ('训练数据目录:%s'% data_path)
        logtxt.append ('预训练模型目录:%s'%bert_path)
        logtxt.append ('预加载模型:%s, %s'% (preload, model_outpath))
        logtxt.append ('训练轮数:%d'% epochs)
        logtxt.append ('学习率:%f'% learning_rate)
        logtxt.append ('batch_size:%d'% batch_size)
        
        savetofile('\n'.join(logtxt),  os.path.join(model_outpath, logfile) ) 

    if task == 'train':
        save_run()
        if preload_model != '':
            model.load_weights(preload_model)
            print('原模型权重已加载。')

        evaluator = Evaluator()

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
        print('正在预测测试集数据...')
        # 加载模型权重  
        model.load_weights(model_file_weight)

        # 加载待预测数据
        pred_file = args.pred_file
        pred_outfile = args.pred_outfile
        if pred_file != '' and pred_outfile != '':
            predict_test(model, pred_file, pred_outfile)

    if task in ['train', 'eval']:
        y_true = np.array([b for a,b in valid_data])

        # 加载验证集
        if 1 or len(test_data) == 0:
            data_generator = data_generator(valid_data, batch_size)
        else:
            data_generator = data_generator(test_data, batch_size)

        # 加载模型
        print('正在加载模型...')
        model.load_weights(model_file_weight)

        print('正在验证数据集...')
        pred = model.predict_generator(data_generator.forfit(random=False), 
                        steps=len(data_generator), verbose=1)

        y_pred = np.array([pred[0].argmax(axis=1), pred[1].argmax(axis=1)])
        y_pred = y_pred.T

        # 计算多任务模型中各个任务的F1值
        multitask_model_score(y_true, y_pred)

    if task == 'online': # 实时预测        
        # 加载label字典
        label_dict = {}
        if label_dict_file:
            label_dict = load_catalog(label_dict_file)
        
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

