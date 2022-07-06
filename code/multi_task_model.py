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
from funlib import *
#from bert4keras.backend import keras, set_gelu, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, DataGenerator, to_array
from keras.layers import Dropout, merge, Input, Lambda
from keras.layers import Lambda, Dense, Bidirectional, LSTM, Concatenate
from keras.utils.np_utils import to_categorical
from keras.models import Model,load_model
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

parser = argparse.ArgumentParser(description='单文本多任务分类模型')
parser.add_argument('--task', type=str, default="train", help='train,eval,predict,online')
parser.add_argument('--data_path', type=str, default="./data/", help='data path')
parser.add_argument('--model_outpath', type=str, default="./models", help='model outpath')
parser.add_argument('--bert_path', type=str, default='', help='bert_path')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size=32')
parser.add_argument('--epochs', type=int, default=10, help='epochs=10')
parser.add_argument('--lr', type=float, default=1e-5, help='learning_rate')
parser.add_argument('--pred_file', type=str, default='', help='预测文件')
parser.add_argument('--pred_outfile', type=str, default='', help='预测输出文件')
parser.add_argument('--preload_model', type=str, default='', help='预加载模型文件')
parser.add_argument('--pred_detail', type=int, default=0, help='pred_detail')
parser.add_argument('--frozen', type=int, default=-1, help='frozen')
parser.add_argument('--debug', type=int, default=0, help='debug')
parser.add_argument('--adv', type=int, default=0, help='启用对抗')
parser.add_argument('--splited', type=int, default=0, help='is_split')

args = parser.parse_args()
task = args.task
epochs = args.epochs
learning_rate = args.lr
batch_size = args.batch_size
data_path = args.data_path
model_outpath = args.model_outpath
bert_path = args.bert_path
#label_dict_file = args.label_dict
preload_model = args.preload_model
label_dict_file = ''
debug = args.debug
frozen = args.frozen
adv = args.adv
splited = args.splited

maxlen = 256
set_gelu('tanh')  # 切换gelu版本

np.random.seed(1234)
tf.set_random_seed(1234)


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

class data_generator_m(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            #token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            token_ids, segment_ids = list_encode(tokenizer, text, maxlen)
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


# 多任务模型
def MModel(config_path, checkpoint_path, num_classes):
    bert = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        return_keras_model=False,
    )
    #output = Lambda(lambda x: K.mean(x, axis=1), name='MEAN-token')(bert.model.output)
    output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)

    out = []
    for i,num in enumerate(num_classes):
        tmp = Dense(units=num, activation='softmax', name='out_%d'%i)(output)
        out.append(tmp)
    '''
    out_0 = Dense(units=num_classes[0], activation='softmax', name='out_0')(output)
    out_1 = Dense(units=num_classes[1], activation='softmax', name='out_1')(output)
    out = [out_0, out_1]
    '''

    model = Model(bert.model.input, out, name="MModel")
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
    print('冻结层:%s'% layer_name) 

# sys.exit()


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
    metrics=['acc'],    #, f1sum
)

# 写好函数后，启用对抗训练只需要一行代码
if adv == 1:
    adversarial_training(model, 'Embedding-Token', 0.5)

if splited == 1:
    loader = load_data_m
    loader_all = load_all_data_m
    generator = data_generator_m
else:
    loader = load_data
    loader_all = load_all_data
    generator = data_generator

# 加载数据集
train_data, valid_data, test_data = loader_all(data_path)

# 转换数据集
train_generator = generator(train_data, batch_size)
valid_generator = generator(valid_data, batch_size)
test_generator = generator(test_data, batch_size)
  
'''
print('batch_size:', batch_size)
for txt, label in train_generator.forfit():
    print('label.shape:', label.shape)
    print(label)
    break;
sys.exit()
'''

def evaluate(data):
    y_true = np.array([b for a,b in data])
    t_generator = generator(data, batch_size)

    pred = model.predict_generator(t_generator.forfit(random=False), 
                    steps=len(t_generator), verbose=0)

    #y_pred = np.array([pred[0].argmax(axis=1), pred[1].argmax(axis=1)])
    #y_pred = y_pred.T
    f1 = multitask_f1(y_true, pred)
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
def predict_test(model, pred_file, outfile, pred_detail):

    '''
    # 自动生成输出文件名
    if outfile == "":
        fname = 'submit_%s.tsv' % time.strftime('%Y%m%d_%H%M%S', time.localtime())
        p, f = os.path.split(pred_file)
        outfile = os.path.join(p, fname)
    '''

    # 加载数据
    pred_data = loader(pred_file)
    print('pred_data:', len(pred_data))
    pred_generator = generator(pred_data, batch_size)
    
    # 批量预测
    pred = model.predict_generator(pred_generator.forfit(random=False), steps=len(pred_generator), verbose=1)

    y_pred = [pred[0].argmax(axis=1), pred[1].argmax(axis=1)]
    #print('pred[0]:', pred[0].shape, type(pred[0]))
    #print('y_pred[0]:', y_pred[0])

    # 是否输出概率：pred_detail
    if pred_detail==1:
        ll = np.arange(pred[0].shape[0]) 
        y_prob = [pred[0][ll, y_pred[0]], pred[1][ll, y_pred[1]]]
        #print('y_prob[0]:', y_prob[0])
        #sys.exit()

        df = pd.DataFrame({"id": range(len(y_pred[0])), 
                        "label_i":y_pred[0], "label_j":y_pred[1],
                        "prob_i":y_prob[0], "prob_j":y_prob[1],
                        })
    else:
        df = pd.DataFrame({"id": range(len(y_pred[0])), "label_i":y_pred[0], "label_j":y_pred[1]})
    
    # 保存预测的提交结果
    df.to_csv(outfile, index=0)
    print('提交文件已生成：%s'%outfile)


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
        '''
        # Checkpoint the weights when validation accuracy improves
        '''
        filepath = os.path.join(model_outpath, 'best_model.h5')
        checkpoint= ModelCheckpoint(filepath, 
                    save_weights_only=False,
                    save_best_only=True, 
                    monitor='f1sum', #multitask_f1  val_acc
                    mode='max',
                    verbose=1, 
                    )

        # 训练模型
        print('开始训练模型...')
        history_fit = model.fit(
            train_generator.forfit(),
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            #validation_data=valid_generator.forfit(),
            #validation_steps=len(valid_generator),
            #callbacks=[evaluator]
            callbacks=[evaluator, checkpoint],
            shuffle=False,
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
        pred_detail = args.pred_detail
        # 默认的数据
        if pred_file == '':
            pred_file = os.path.join(data_path, 'test.tsv')
        if pred_outfile == '':
            pred_outfile = os.path.join(model_outpath, 'submit.csv')

        if pred_file != '' and pred_outfile != '':
            predict_test(model, pred_file, pred_outfile, pred_detail)

    if task in ['train', 'eval']:
        y_true = np.array([b for a,b in valid_data])

        # 加载验证集
        t_generator = generator(valid_data, batch_size)

        # 加载模型
        print('正在加载模型...')
        model.load_weights(model_file_weight)

        print('正在验证数据集...')
        pred = model.predict_generator(t_generator.forfit(random=False), 
                        steps=len(t_generator), verbose=1)

        # 计算多任务模型中各个任务的F1值
        multitask_f1(y_true, pred, show=1)

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

