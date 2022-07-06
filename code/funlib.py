#!/usr/bin/env python3
#coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'

'''
通用库
'''
import os
import sys
import numpy as np
from bert4keras.backend import keras, search_layer, K, set_gelu

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
    return D#[:16]

def load_data_m(filename):
    """加载数据 多列方式加载: 5列数据+2列标签
    """
    D = []
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.read().split('\n')
        for l in data:
            text = []
            x = l.strip().split('\t')
            if len(x)>=7:
                text, (label1, label2) = x[:5], x[5:]
            if text:
                label = [int(label1), int(label2)]
                D.append((text, label))
    return D[:64]

def load_all_data(path):
    print('正在加载数据集...')
    train_data = load_data(os.path.join(path, 'train.tsv'))
    valid_data = load_data(os.path.join(path, 'dev.tsv'))
    test_data = load_data(os.path.join(path, 'test.tsv'))
   
    return train_data,valid_data,test_data

def load_all_data_m(path):
    print('正在加载数据集...')
    train_data = load_data_m(os.path.join(path, 'train.tsv'))
    valid_data = load_data_m(os.path.join(path, 'dev.tsv'))
    test_data = load_data_m(os.path.join(path, 'test.tsv'))
   
    return train_data,valid_data,test_data


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

def list_encode(tokenizer, txt_list:list, maxlen):
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

def adversarial_training(model, embedding_name, epsilon=1):
    """给模型添加对抗训练
    其中model是需要添加对抗训练的keras模型，embedding_name
    则是model里边Embedding层的名字。要在模型compile之后使用。
    """
    if model.train_function is None:  # 如果还没有训练函数
        model._make_train_function()  # 手动make
    old_train_function = model.train_function  # 备份旧的训练函数

    # 查找Embedding层
    for output in model.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break
    if embedding_layer is None:
        raise Exception('Embedding layer not found')

    # 求Embedding梯度
    embeddings = embedding_layer.embeddings  # Embedding矩阵
    gradients = K.gradients(model.total_loss, [embeddings])  # Embedding梯度
    gradients = K.zeros_like(embeddings) + gradients[0]  # 转为dense tensor

    # 封装为函数
    inputs = (
        model._feed_inputs + model._feed_targets + model._feed_sample_weights
    )  # 所有输入层
    embedding_gradients = K.function(
        inputs=inputs,
        outputs=[gradients],
        name='embedding_gradients',
    )  # 封装为函数

    def train_function(inputs):  # 重新定义训练函数
        grads = embedding_gradients(inputs)[0]  # Embedding梯度
        delta = epsilon * grads / (np.sqrt((grads**2).sum()) + 1e-8)  # 计算扰动
        K.set_value(embeddings, K.eval(embeddings) + delta)  # 注入扰动
        outputs = old_train_function(inputs)  # 梯度下降
        K.set_value(embeddings, K.eval(embeddings) - delta)  # 删除扰动
        return outputs

    model.train_function = train_function  # 覆盖原训练函数


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

def f1sum(y_true, pred):
    print('-'*40)

    print('y_true:', type(y_true))
    print('pred:', type(pred))
    print('pred[0]:', type(pred[0]))
    #yy = K.eval(pred[0])
    yy = pred[0].numpy()
    print('yy:', yy)

    #y_pred = K.cast(pred, 'float32')
    y_pred = K.argmax(pred[0], axis=1)

    print('y_pred:', type(y_pred))
    print('y_pred:', y_pred)
    #ret = multitask_f1(y_true, y_pred)

    # 多少个任务
    task_count = y_true.shape[1]
    print('task_count:', task_count)
    f1_final = 0
    for i in range(task_count):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]
        if i==0:
            f1 = f1_score(y_t, y_p, average='macro')  #weighted  macro micro)
        else:
            f1 = f1_score(y_t, y_p, average='micro')  #weighted  macro micro)
        f1_final += f1

    return f1_final

def multitask_f1(y_true, pred, show=0):
    ''' 
    多任务模型的模型评价 用F1之和作为指标
    '''
    from sklearn.metrics import f1_score, accuracy_score, recall_score
    

    y_pred = np.array([pred[0].argmax(axis=1), pred[1].argmax(axis=1)])
    y_pred = y_pred.T

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
    pass

