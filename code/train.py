#!/usr/bin/env python3
#coding:utf-8

import numpy as np
import torch
from sklearn import metrics
from transformers import get_cosine_schedule_with_warmup
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import timedelta
import time
import pickle as pkl
# from pytorch_pretrained.optimization import BertAdam
from torch.optim import Adam

from dataset import *


plt.rcParams['font.family'] = 'SimHei'
model_path = 'roberta_lstm/roberta_multi.ckpt'
class_cat = ['乳腺外科', '产前检查', '内科', '呼吸内科', '咽喉疾病', '妇产科', '小儿保健', '小儿呼吸系统疾病', '小儿消化疾病','小儿耳鼻喉',
             '心内科', '消化内科', '甲状腺疾病', '皮肤科', '直肠肛管疾病', '眼科', '神经内科', '脊柱退行性变', '运动医学', '骨科']
class_label = ['乳房囊肿', '乳腺增生','乳腺疾病', '乳腺肿瘤','产前检查', '儿童保健', '先兆流产', '内科其他', '剖腹产', '发育迟缓','呼吸内科其他',
               '咽喉疾病', '喉疾病', '围产保健', '外阴疾病', '妇科病', '宫腔镜', '小儿呼吸系统疾病', '小儿咳嗽', '小儿感冒', '小儿支气管炎',
               '小儿支气管肺炎', '小儿消化不良', '小儿消化疾病', '小儿耳鼻喉其他', '小儿肺炎', '心内科其他', '心脏病', '扁桃体炎', '早孕反应',
               '月经失调', '桥本甲状腺炎', '消化不良', '消化内科其他' ,'消化道出血', '甲减', '甲状腺功能异常', '甲状腺疾病', '甲状腺瘤',
               '甲状腺结节', '痔疮', '皮肤病', '皮肤瘙痒', '皮肤科其他', '直肠肛管疾病', '眼部疾病', '神经内科其他', '微量元素缺乏', '羊水异常',
               '肺部疾病', '胃病', '脊柱退行性变', '腰椎间盘突出', '腹泻', '腹痛', '膝关节半月板损伤', '膝关节损伤', '膝关节韧带损伤', '运动医学',
               '韧带损伤', '骨科其他']


def get_time_dif(start_time):
    end_time = time.time()
    time_idf = end_time - start_time
    return timedelta(seconds=int(round(time_idf)))


def train(model, train_iter, val_iter, epoch):
    start_time = time.time()
    # facal_loss = torch.nn.CrossEntropyLoss()
    # 启动BathNormalization和dropout
    model.train()
    # 拿到所有model的参数
    parm_optimizer = list(model.named_parameters())
    # 不需要衰减的参数
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # any:表示列表中都不为空或0,只要有一个n参数在no_decay中就不需要进行参数衰减.
    optimizer_grouped_parameters = [
        {'params': [p for n, p in parm_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in parm_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = Adam(params=optimizer_grouped_parameters, lr=1e-5,)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * len(train_iter),
                                                num_training_steps=10 * len(train_iter))

    total_batch = 0
    dev_best_acc = float('-inf')
    # last_improve = 0
    # flag = False
    train_loss_list = []
    train_acc_list = []
    batch_list = []
    fn_loss = torch.nn.CrossEntropyLoss()
    model.train()
    val_acc = []
    val_loss = []
    for i in range(epoch):
        print('Epoch[{}/{}]'.format(i+1, epoch))
        for i, (content, labels) in enumerate(train_iter):
            cat_id_out, label_out = model(content)
            model.zero_grad()
            cat_id = labels[0]
            label = labels[1]
            loss_1 = fn_loss(cat_id_out, cat_id)
            loss_2 = fn_loss(label_out, label)
            loss = loss_1 + loss_2
            loss.backward()
            optimizer.step()
            scheduler.step()
            cat_id = cat_id.data.cpu()
            label = label.data.cpu()
            train_loss_list.append(loss.item())
            predict_1 = torch.max(cat_id_out, 1)[1].cpu()
            predict_2 = torch.max(label_out, 1)[1].cpu()
            acc_1 = metrics.accuracy_score(cat_id, predict_1)
            acc_2 = metrics.accuracy_score(label, predict_2)
            acc = (acc_1 + acc_2) / 2
            train_acc_list.append(acc)
            batch_list.append(total_batch)

            if total_batch % 100 == 0:
                # 训练集没进行100次, 进行一次验证
                dev_acc, dev_loss = evaluate(model, val_iter)
                val_loss.append(dev_loss)
                val_acc.append(dev_acc)
                if dev_acc > dev_best_acc:
                    dev_best_acc = dev_acc
                    torch.save(model.state_dict(), model_path)
                    improve = '*'
                    # last_improve = total_batch
                else:
                    improve = '~'
                time_idf = get_time_dif(start_time)
                msg = 'Iter:{0:6}, Train Loss:{1:5.2}, Train Acc:{2:6.2},' \
                      'Val Loss:{3:5.2}, Val Acc:{4:6.2}, time:{5}{6},cat_loss:{7:5.2},label_loss:{8:5.2}, cat_acc:{9:6.2}, label_acc:{10:6.2}'
                print(msg.format(total_batch, loss.item(), acc, dev_loss, dev_acc, time_idf, improve, loss_1.item(), loss_2.item(), acc_1, acc_2))
                model.train()
            total_batch += 1
            # if total_batch - last_improve > 1000:
            #     print('模型在验证集上很久没有提升了, 模型终止训练')
            #     flag = True
            #     break
        # if flag:
        #     break
    fig = plt.figure(num=1, figsize=(12, 9))
    ax1 = fig.add_subplot(211)
    ax1.plot(batch_list, train_loss_list)
    plt.title('训练损失函数')
    plt.xlabel("训练次数")
    plt.ylabel('训练损失')
    ax2 = fig.add_subplot(212)
    ax2.plot(batch_list, train_acc_list)
    plt.title('训练准确率函数')
    plt.xlabel("训练次数")
    plt.ylabel('训练准确率')
    plt.show()

def evaluate(model, val_iter):
    """
    模型验证函数
    """
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for contents, labels in val_iter:
            cat_id_out, label_out = model(contents)
            cat_id = labels[1]
            label = labels[0]
            loss_1 = torch.nn.CrossEntropyLoss()(cat_id_out, cat_id)
            loss_2 = torch.nn.CrossEntropyLoss()(label_out, label)
            loss_total += loss_1 + loss_2
            cat_id = cat_id.data.cpu().numpy()
            label = label.data.cpu().numpy()
            pred_1 = torch.max(cat_id_out.data, 1)[1].cpu().numpy()
            pred_2 = torch.max(label_out.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, cat_id)
            labels_all = np.append(labels_all, label)
            predict_all = np.append(predict_all, pred_1)
            predict_all = np.append(predict_all, pred_2)
        acc = metrics.accuracy_score(labels_all, predict_all)

    return acc, loss_total/len(val_iter)

# def test(model, test_iter):
#     """
#     模型的测试
#     """
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
#     loss_total = 0
#     predict_cat_all = np.array([], dtype=int)
#     predict_label_all = np.array([], dtype=int)
#     labels_cat_all = np.array([], dtype=int)
#     labels_label_all = np.array([], dtype=int)
#     with torch.no_grad():
#         for idx, (content, labels) in enumerate(test_iter):
#             cat_id_out, label_out = model(content)
#             facal_loss = nn.CrossEntropyLoss()
#             cat_id = labels[1]
#             label = labels[0]
#             loss1 = facal_loss(cat_id_out, cat_id)
#             loss2 = facal_loss(label_out, label)
#             loss = loss1 + loss2
#             loss_total += loss.item()
#             cat_id = cat_id.data.cpu().numpy()
#             label = label.data.cpu().numpy()
#             predict_cat_id = torch.max(cat_id_out.data, 1)[1].cpu().numpy()
#             predict_label = torch.max(label_out.data, 1)[1].cpu().numpy()
#             # labels_all = np.append(labels_all, cat_id)
#             # labels_all = np.append(labels_all, label)
#             # predict_all = np.append(predict_all, predict_cat_id)
#             # predict_all = np.append(predict_all, predict_label)
#             predict_cat_all = np.append(predict_cat_all, predict_cat_id)
#             predict_label_all = np.append(predict_label_all, predict_label)
#             labels_cat_all = np.append(labels_cat_all, cat_id)
#             labels_label_all = np.append(labels_label_all, label)
#
#         acc_cat = metrics.accuracy_score(predict_cat_all, labels_cat_all)
#         acc_label = metrics.accuracy_score(predict_label_all, labels_label_all)
#
#         # report:精确率, 准确率, 召回率
#         report_cat = metrics.classification_report(predict_cat_all, labels_cat_all, target_names=class_cat, digits=4)
#         report_label = metrics.classification_report(predict_label_all, labels_label_all,
#                                                      target_names=class_label, digits=4)
#
#         confusion_cat = metrics.confusion_matrix(predict_cat_all, labels_cat_all)  # 混淆矩阵
#         confusion_label = metrics.confusion_matrix(predict_label_all, labels_label_all)
#     return acc_cat, acc_label, loss_total/len(test_iter), report_cat, report_label, confusion_cat, confusion_label


if __name__ == '__main__':
    from model import Model
    #from dataset import get_loader
    #from dataset import Data
    #from dataset import collate_fn

    train_loader, dev_loader = get_loader()
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #roberta_path = 'roberta_pretrain'
    print('bert_path:', bert_path)
    #roberta_path = '/mnt/sda1/models/roberta-base_pytorch'
    #roberta_path = '/mnt/sda1/models/chinese_wwm_pytorch'

    model = Model(bert_path).to(device)
    train(model, train_loader, dev_loader, 5)
    # acc_cat, acc_label, loss, report_cat, report_label, confusion_cat, confusion_label = test(model, test_loader)
    # print('acc_cat:', acc_cat)
    # print('acc_label:', acc_label)
    # print('loss:', loss)
    # print('report_cat:', report_cat)
    # print('report_label:', report_label)
    # print('confusion_cat:', confusion_cat)
    # print('confusion_label:', confusion_label)
