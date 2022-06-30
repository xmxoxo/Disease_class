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
#from bert4keras.tokenizers import SpTokenizer
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, DataGenerator, to_array
#from bert4keras.snippets import open, ViterbiDecoder, to_array
#from bert4keras.layers import ConditionalRandomField
from keras.layers import Dropout, merge, Input, Lambda
from keras.layers import Lambda, Dense, Bidirectional, LSTM, Concatenate
from keras.utils.np_utils import to_categorical
from keras.models import Model,load_model

parser = argparse.ArgumentParser(description='单文本多任务分类模型')
parser.add_argument('--task', type=str, default="train", help='train,eval,predict,online')
parser.add_argument('--data_path', type=str, default="./data/", help='data path')
parser.add_argument('--model_outpath', type=str, default="./models/", help='model outpath')
parser.add_argument('--bert_path', type=str, default='', help='bert_path')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size=32')
parser.add_argument('--epochs', type=int, default=10, help='epochs=10')
parser.add_argument('--lr', type=float, default=1e-5, help='learning_rate')
#parser.add_argument('--label_dict', type=str, default='', help='分类字典')
parser.add_argument('--pred_file', type=str, default='data/test.tsv', help='预测文件')
parser.add_argument('--preload_model', type=str, default='', help='预加载模型文件')
parser.add_argument('--debug', type=int, default=0, help='debug')

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
    单条格式：(文本+TAB+标签id)
    输出格式：(文本, 标签id)
    """
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.read().split('\n')
        for l in data:
            x = l.strip().split('\t')
            if len(x)==1:
                text, label1, label2 = x[0], 0, 0
            if len(x)>=3:
                text, label1, label2 = x[:3]
            if text:
                label = [int(label1), int(label2)]

                '''
                Y0 = to_categorical(int(label1), 20)
                Y1 = to_categorical(int(label2), 61)
                label = np.concatenate((Y0, Y1), axis=0).tolist()
                '''
                D.append((text, label))
    return D#[:32]


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
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(label)
            
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                
                batch_labels = [batch_labels[:,0], batch_labels[:,1]]
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

# 多任务模型
def MModel(config_path, checkpoint_path, num_classes):
    bert = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        return_keras_model=False,
    )
    #output = Lambda(lambda x: K.mean(x, axis=1), name='MEAN-token')(bert.model.output)
    output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)

    '''
    out = []
    for i,num in enumerate(num_classes):
        tmp = Dense(units=num, activation='softmax', name='out_%d'%i)(output)
        out.append(tmp)
    '''
    out_0 = Dense(units=num_classes[0], activation='softmax', name='out_0')(output)
    out_1 = Dense(units=num_classes[1], activation='softmax', name='out_1')(output)
    out = [out_0, out_1]

    # out = Concatenate(axis=1)(out)
    model = Model(bert.model.input, out, name="MModel")
    return model

print('正在创建模型...')
# 分类大小
num_classes = [20, 61]
model = MModel(config_path, checkpoint_path, num_classes)
# model.summary()
print('model output:', model.output)
#sys.exit()

if frozen == 1:
    # 冻住指定的out_1层 禁止训练
    layer_out_1 = model.get_layer('out_1')
    layer_out_1.trainable = False

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
    total, right = 0., 0.
    y_trues = []
    y_preds = []
    for x_true, y_true in data:
        y_pred = model.predict(x_true)
        #print('y_pred=', y_pred)
        y_pred = np.array([y_pred[0].argmax(axis=1), y_pred[1].argmax(axis=1)])
        y_preds.append(y_pred)
        y_pred = np.array(list(zip(*y_pred)))
        #y_pred = list(zip(*y_pred))
        #print('y_pred=', y_pred)
        
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
def predict_test(model, datfile, outfile, label_dict=None):
    # 加载数据
    #pred_data = load_data(datfile)
    pred_data = load_pred_data(datfile)
    print('pred_data:', len(pred_data))

    pred_generator = data_generator(pred_data, batch_size)
    
    # 批量预测
    pred = model.predict_generator(pred_generator.forfit(random=False), steps=len(pred_generator), verbose=1)
    y_pred = np.argmax(pred, axis=1)
    if label_dict:
        y_label = [label_dict[x] for x in y_pred]

    # 连接原数据
    #text = [x for x, lbl in pred_data]
    text, label = list(zip(*pred_data))
    df_pred = pd.DataFrame({'title':list(text), 'label': list(label), 'pred_label': y_label})
    
    # 保存
    df_pred.to_csv(outfile, index=0, sep='\t')


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
 
def multitask_model_sclre(y_true, y_pred):
    from sklearn.metrics import f1_score, accuracy_score, recall_score
    # 多少个任务
    task_count = y_true.shape[1]
    f1_final = 0
    for i in range(task_count):
        print( ('Task: %d'%i).center(40, '-'))
        y_t = y_true[:, i]
        y_p = y_pred[:, i]
        acc = accuracy_score(y_t, y_p)
        recall = accuracy_score(y_t, y_p)
        if i==0:
            f1 = f1_score(y_t, y_p, average='macro')  #weighted  macro micro)
            print('Accuracy:%.2f Recall:%.2f F1-macro:%.2f' % (acc, recall, f1))
        else:
            f1 = f1_score(y_t, y_p, average='micro')  #weighted  macro micro)
            print('Accuracy:%.2f Recall:%.2f F1-micro:%.2f' % (acc, recall, f1))
            
        f1_final += f1

    print('F1_total:%.4f' % f1_final)
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
        if preload==1:
            model.load_weights(preload_model)
            print('模型已加载。')

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
        # 批量预测
        print('正在预测测试集数据...')
        label_dict = {}
        if label_dict_file:
            label_dict = load_catalog(label_dict_file)

        # 加载数据
        # train_data, valid_data, test_data = load_all_data(data_path)
        
        # 加载模型 
        model.load_weights(model_file_weight)

        '''
        # 加载待预测数据, 格式：一行一条数据 
        pred_file = args.pred_file
        # 生成输出文件名
        fname = 'predict_%s.tsv' % time.strftime('%Y%m%d_%H%M%S', time.localtime())
        data_path = os.path.split(pred_file)[0]
        outfile = os.path.join(data_path, fname)
        
        predict_test(model, pred_file, outfile, label_dict=label_dict)
        print('预测结果已保存:%s' % outfile)
        '''
        pred = model.predict_generator(test_generator.forfit(random=False), 
                                        steps=len(test_generator), verbose=1)

        y_pred = [pred[0].argmax(axis=1), pred[1].argmax(axis=1)]

        # 保存预测的提交结果
        df = pd.DataFrame({"id": range(len(y_pred[0])), "label_i":y_pred[0], "label_j":y_pred[1]})
        subfile = os.path.join(model_outpath, 'submit.csv')
        df.to_csv(subfile, index=0)
        print('提交文件已生成：%s'%subfile)

    
    if task in ['train', 'eval']:
        # 加载数据
        #if task == 'eval':
        #    train_data, valid_data, test_data = load_all_data(data_path)
        y_true = np.array([x[1] for x in valid_data])
        '''
        print('y_true.shape:', y_true.shape)
        print(y_true)
        print('-'*40)
        #sys.exit()
        '''

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
        '''
        print('y_pred.shape:', y_pred.shape)
        print(y_pred)
        print('-'*40)
        '''

        # 计算多任务模型中各个任务的F1值
        multitask_model_sclre(y_true, y_pred)


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

