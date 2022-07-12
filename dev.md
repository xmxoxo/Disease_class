# 非标准化疾病诉求的简单分诊挑战赛2.0

http://challenge.xfyun.cn/topic/info?type=disease-claims-2022&option=ssgy

## 基础思路 


数据总共有以下几个字段：

`id	age	diseaseName	conditionDesc	title	hopeHelp`

预测字段两个：`label_i, label_j`

直接把原始数据字段：`age，diseaseName，conditionDesc，title，hopeHelp`

用分号全部串起来，作为文本；

然后当作两个分类任务，直接预测目标字段；

数据简单处理：训练数据随机打乱，然后拆分成8:2；


数据几个指标情况:
	各文本字段的最大长度；
	各分类的占比情况；

模型参考：

```
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
    for num in num_classes:
        tmp = Dense(units=num, activation='softmax')(output)
        out.append(tmp)

    model = Model(bert.model.input, out, name="MModel")
    return model

# 模型目录
bert_path = r'F:\models\chinese_L-12_H-768_A-12'
config_path = os.path.join(bert_path, 'bert_config.json')
checkpoint_path = os.path.join(bert_path, 'bert_model.ckpt')
dict_path = os.path.join(bert_path, 'vocab.txt')

# 分类大小
num_classes = [20,62]

创建模型
model = MModel(config_path, checkpoint_path, num_classes)
```

## 数据处理

分类`label_j`的标签值是从-1到60，全部+1，转换成0到61，共62个；

```
python data_process.py --task=data_trans
```

生成的数据会保存到`data` 目录下；


## 模型训练

bert4keras架构的多任务模型

```
python multi_task_model.py --task=train --epochs=1 --bert_path=预训练目录 --batch_size=32
```

预测数据生成提交文件：

```
python multi_task_model.py --task=predict
```

生成的提交文件在：`models/submit.csv`

线上提交后得分为：`0.88533`


在验证数据集上进行验证并计算指标：

```
python multi_task_model.py --task=eval
```

在验证集上验证的结果：
```
143/143 [==============================] - 15s 106ms/step
----------------Task: 0-----------------
Accuracy:0.89 Recall:0.89 F1-macro:0.88
----------------Task: 1-----------------
Accuracy:0.66 Recall:0.66 F1-micro:0.66
F1_total:1.5430
```

结果分析：
线上得分非常低，都没有超过1；分析了提交的数据：
```
python data_process.py --task=submit_test
```

可以看到提交的结果中,`label_j`这个任务里，分类为0-61，总数是62个；
而题目中说明的是:

```
以及就诊方向标签label_i∈int [0,19]，
疾病方向标签label_j∈int [0,60] 
(训练数据中疾病方向标签存在缺失，以-1标记，测试数据中没有)。
```
说明 'label_j'这个任务正确的输出应该是0-60总共61个分类；
对于数据集中标有-1的样本，不应该转换标签，而是应该删除或者做其它处理。

Todo: 先把训练集中`label_j`列为-1的样本删除，重新训练模型，重新提交预测结果；

-----------------------------------------

##  关于-1样本的处理思路 

在label_j列中，有一些样本标的是-1, 这是表示有一些样本中并没有对label_j进行标注， 
所以这个-1不能当作分类的标签，而是要重新考虑处理方法。

数据情况：

    原始数据条数：22868

    -1标签样本条数：8746

    完整样本条数： 14122

首先确定：模型的label_j任务，输出应为0-60,共61类；

思路一：**伪标签法**

* 第一步：

	对于-1标签的样本，先从数据集中拆分出来，称为“-1样本集”，

	把剩下的完整数据集拿去训练模型，得到模型A；

* 第二步：

	使用训练好的模型A对“-1样本集”进行预测，模型预测了label_i和label_j，

	但是把预测出来的 label_i 丢弃掉；

 	将“-1样本集”中的label_i和预测结果label_j合并，得到 “伪标签样本集”

* 第三步：

	在原来模型A的基础上，继续用“伪标签样本集”进行训练，得到最终模型B

	模型B即为最终得到的模型，用它预测测试集并提交；


思路二： **冻层法**

* 第一步：（与 伪标签法 第一步相同）

	对于-1标签的样本，先从数据集中拆分出来，称为“-1样本集”，

	把剩下的完整数据集拿去训练模型，得到模型A；

* 第二步：

	加载模型A权重， 冻结label_j模型层，使用 “-1样本集” 对模型进行继续训练， 得到最终模型B



对比后感觉还是 **伪标签法** 比较好，对label_j再做了一次训练；


## 伪标签法训练

重新处理数据：
```
python data_process.py --task=data_trans
```

重新生成训练文件以及"-1标签集"数据文件：`label_j.tsv`

训练模型A：

```
python multi_task_model.py --task=train --epochs=30 --batch_size=48 --model_outpath=model_a
```

训练过程：
```
Epoch 24/30
236/236 [==============================] - 107s 454ms/step - loss: 0.2502 - out_0_loss: 0.0510 - out_1_loss: 0.1992 - out_0_acc: 0.9880 - out_1_acc: 0.9415
val_acc: 0.76637, best_val_acc: 0.76885

Epoch 25/30
236/236 [==============================] - 108s 456ms/step - loss: 0.2389 - out_0_loss: 0.0477 - out_1_loss: 0.1911 - out_0_acc: 0.9884 - out_1_acc: 0.9436
val_acc: 0.76708, best_val_acc: 0.76885

Epoch 26/30
236/236 [==============================] - 106s 451ms/step - loss: 0.2348 - out_0_loss: 0.0485 - out_1_loss: 0.1863 - out_0_acc: 0.9884 - out_1_acc: 0.9437
val_acc: 0.76602, best_val_acc: 0.76885

Epoch 27/30
236/236 [==============================] - 107s 452ms/step - loss: 0.2212 - out_0_loss: 0.0425 - out_1_loss: 0.1787 - out_0_acc: 0.9912 - out_1_acc: 0.9472
val_acc: 0.76566, best_val_acc: 0.76885

Epoch 28/30
236/236 [==============================] - 107s 454ms/step - loss: 0.2119 - out_0_loss: 0.0407 - out_1_loss: 0.1712 - out_0_acc: 0.9912 - out_1_acc: 0.9512
val_acc: 0.76425, best_val_acc: 0.76885

Epoch 29/30
236/236 [==============================] - 107s 453ms/step - loss: 0.2010 - out_0_loss: 0.0394 - out_1_loss: 0.1615 - out_0_acc: 0.9908 - out_1_acc: 0.9548
val_acc: 0.76531, best_val_acc: 0.76885

Epoch 30/30
236/236 [==============================] - 106s 451ms/step - loss: 0.2013 - out_0_loss: 0.0400 - out_1_loss: 0.1613 - out_0_acc: 0.9904 - out_1_acc: 0.9548
val_acc: 0.76673, best_val_acc: 0.76885

正在保存训练数据...
训练曲线图已保存。
正在预测测试集数据...
159/159 [==============================] - 23s 146ms/step
提交文件已生成：model_a/submit.csv
正在加载模型...
正在验证数据集...
59/59 [==============================] - 8s 144ms/step
----------------Task: 0-----------------
Accuracy:0.88 Recall:0.88 F1-macro:0.88
----------------Task: 1-----------------
Accuracy:0.79 Recall:0.79 F1-micro:0.79
F1_total:1.6621
```

线上得分：1.64699

训练完成后，预测-1标签数据集：
```
python multi_task_model.py --task=predict --model_outpath=model_a --pred_file=data/label_j.tsv --pred_outfile=model_a/pred_label_j.csv
```

运行结果：
```
274/274 [==============================] - 28s 104ms/step
提交文件已生成：model_a/pred_label_j.csv
```

合并处理预测的-1标签数据集：

```
python data_process.py --task=merge_predict --fname=data/label_j.tsv,model_a/pred_label_j.csv --outpath=data/merge
```
运行后会生成一个新的训练集目录：`data/merge`，目录里包含了合并后的-1标签数据作为训练集，
以及原有的验证集和测试集；


第三步：在原模型基础上训练合并数据集，得到模型B

代码优化： 验证过程使用比赛指定的F1_final指标作为评估指标，保证模型的F1值最高；


```
python multi_task_model.py --task=train --epochs=30 --batch_size=48 \
--data_path=data/merge \
--model_outpath=model_b \
--preload_model=model_a/model.weights \
--pred_outfile=model_b/submit.csv
```

win下命令行：
```
python multi_task_model.py --task=train --epochs=30 --batch_size=48 --data_path=data/merge --model_outpath=model_b --preload_model=model_a/model.weights --pred_outfile=model_b/submit.csv
```

训练结果：

```
Epoch 1/5
2022-07-01 11:41:36.359858: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.11
183/183 [==============================] - 90s 491ms/step - loss: 0.8691 - out_0_loss: 0.4841 - out_1_loss: 0.3850 - out_0_acc: 0.8566 - out_1_acc: 0.8774
model score: [(0.8817699115044247, 0.8817699115044247, 0.8758975364712057), (0.7847787610619469, 0.7847787610619469, 0.784778761061947)] F1_final:1.6607
val_F1: 1.66068, best_val_F1: 1.66068

Epoch 2/5
183/183 [==============================] - 82s 446ms/step - loss: 0.8063 - out_0_loss: 0.4313 - out_1_loss: 0.3750 - out_0_acc: 0.8702 - out_1_acc: 0.8758
model score: [(0.8835398230088496, 0.8835398230088496, 0.8763207095501098), (0.783362831858407, 0.783362831858407, 0.783362831858407)] F1_final:1.6597
val_F1: 1.65968, best_val_F1: 1.66068

Epoch 3/5
183/183 [==============================] - 81s 444ms/step - loss: 0.7324 - out_0_loss: 0.3768 - out_1_loss: 0.3556 - out_0_acc: 0.8866 - out_1_acc: 0.8825
model score: [(0.879646017699115, 0.879646017699115, 0.8708834438996819), (0.7819469026548672, 0.7819469026548672, 0.7819469026548672)] F1_final:1.6528
val_F1: 1.65283, best_val_F1: 1.66068

Epoch 4/5
183/183 [==============================] - 82s 446ms/step - loss: 0.6772 - out_0_loss: 0.3348 - out_1_loss: 0.3424 - out_0_acc: 0.8973 - out_1_acc: 0.8879
model score: [(0.8725663716814159, 0.8725663716814159, 0.8613198805545794), (0.7805309734513274, 0.7805309734513274, 0.7805309734513275)] F1_final:1.6419
val_F1: 1.64185, best_val_F1: 1.66068

Epoch 5/5
183/183 [==============================] - 83s 453ms/step - loss: 0.6353 - out_0_loss: 0.3033 - out_1_loss: 0.3320 - out_0_acc: 0.9058 - out_1_acc: 0.8925
model score: [(0.8693805309734514, 0.8693805309734514, 0.8567172882743102), (0.7776991150442478, 0.7776991150442478, 0.7776991150442479)] F1_final:1.6344
val_F1: 1.63442, best_val_F1: 1.66068

正在保存训练数据...
训练曲线图已保存。
正在预测测试集数据...
pred_data: 7596
159/159 [==============================] - 23s 144ms/step
提交文件已生成：model_b/submit.csv
正在加载模型...
正在验证数据集...
59/59 [==============================] - 8s 142ms/step
----------------Task: 0-----------------
Accuracy:0.88 Recall:0.88 F1-macro:0.88
----------------Task: 1-----------------
Accuracy:0.78 Recall:0.78 F1-micro:0.78
F1_total:1.6607
```


线上提交结果： 1.64455 
训练过程出现过拟合，验证集得分越来越低，线上提交得分也比原来的低。

搜索“伪标签”的资料，需要对原来的思路进行一些调整优化：

思路三：**伪标签法改进**

 第一步：

	使用训练好的模型A对“-1样本集”进行预测，模型预测了label_i和label_j，

	把预测出来的 label_i 丢弃掉，只取出label_j的预测结果中，
	且置信度高于阈值的样本；得到“伪标签数据集”
	数据集中只包含：文本，label_j
	
	这里阈值可以取0.5

* 第二步：

	加载原有模型A的权重，继续用“伪标签样本集”进行训练，
	
	训练时冻结模型label_i模型层，即只训练label_j全连接层，得到最终模型B

	模型B即为最终得到的模型，用它预测测试集并提交；


思路四：**伪标签法改进2**

 第一步：

	使用训练好的模型A对“-1样本集”进行预测，模型预测了label_i和label_j，
	把预测出来的 label_i 丢弃掉，用原始的label_i值替换，
	并过滤 出label_j的预测结果中，置信度高于阈值的样本 得到“伪标签数据集”
	数据集中包含：文本，label_i, label_j
	
	这里阈值可以取0.5

* 第二步：

	将“伪标签样本集”和原始的完整数据集合并，得到“伪标签训练数据集”，
	用这个数据集重新训练一个模型B

	模型B即为最终得到的模型，用它预测测试集并提交；


以上步骤可以重复迭代：把模型B当作模型A继续迭代, 直到模型得分不再提升；

-----------------------------------------

## 新思路：多维度融合

把年龄这个特征单独拿出来，与文本得到的特征进行融合；

```
年龄 ==========>━┓
                  ┣━》全连接层==》输出
文本 ==>BERT ==>━┛
```

数据预处理：

```
python data_process.py --task=data_trans_age
```

数据生成到目录:'./data_age'目录下；

新的模型训练程序：`multi_task_age.py`

独立的试验分支，日志见：[多维度融合试验](dev_age.md)


## 使用大模型及冻层法

训练模型A：

```
python multi_task_model.py --task=train \
--bert_path=/mnt/sda1/models/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16 \
--epochs=30 \
--batch_size=12 \
--lr=1e-5 \
--model_outpath=model_a_wwm
--pred_file=data/test.tsv
--pred_outfile=model_a_wwm/submit.csv 
```

模型大小满足条件：小于4e8

```
Total params: 324,555,857
Trainable params: 324,555,857
```

模型预测:
```
python multi_task_model.py --task=predict \
--bert_path=/mnt/sda1/models/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16 \
--epochs=1 \
--batch_size=12 \
--lr=1e-5 \
--model_outpath=model_a_wwm \
--pred_file=data/test.tsv \
--pred_outfile=model_a_wwm/submit.csv
```

线上得分： 1.67211




训练模型B：
```
python multi_task_model.py --task=train \
--bert_path=/mnt/sda1/models/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16 \
--epochs=3 \
--batch_size=12 \
--data_path=data/merge \
--model_outpath=model_b_wwm \
--preload_model=model_a_wwm/model.weights \
--pred_file=data/test.tsv \
--pred_outfile=model_b_wwm/submit.csv \
--frozen=1
```


训练结果：
```
Epoch 1/3
2022-07-01 15:46:29.102839: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library libcublas.so.11
729/729 [==============================] - 282s 387ms/step - loss: 0.9174 - out_0_loss: 0.3440 - out_1_loss: 0.5734 - out_0_acc: 0.8952 - out_1_acc: 0.8063
model score: [(0.8601769911504424, 0.8601769911504424, 0.8489220866067131), (0.7911504424778761, 0.7911504424778761, 0.7911504424778761)] F1_final:1.6401
val_F1: 1.64007, best_val_F1: 1.64007

Epoch 2/3
729/729 [==============================] - 266s 364ms/step - loss: 0.6221 - out_0_loss: 0.1877 - out_1_loss: 0.4344 - out_0_acc: 0.9367 - out_1_acc: 0.8440
model score: [(0.8460176991150442, 0.8460176991150442, 0.8291330688824334), (0.7706194690265487, 0.7706194690265487, 0.7706194690265487)] F1_final:1.5998
val_F1: 1.59975, best_val_F1: 1.64007

Epoch 3/3
729/729 [==============================] - 267s 366ms/step - loss: 0.3195 - out_0_loss: 0.0999 - out_1_loss: 0.2196 - out_0_acc: 0.9671 - out_1_acc: 0.9259
model score: [(0.8424778761061947, 0.8424778761061947, 0.8229006291297555), (0.7897345132743363, 0.7897345132743363, 0.7897345132743363)] F1_final:1.6126
val_F1: 1.61264, best_val_F1: 1.64007

正在保存训练数据...
训练曲线图已保存。
正在预测测试集数据...
pred_data: 7596
633/633 [==============================] - 62s 97ms/step
提交文件已生成：model_b_wwm/submit.csv
正在加载模型...
正在验证数据集...
236/236 [==============================] - 22s 95ms/step
----------------Task: 0-----------------
Accuracy:0.86 Recall:0.86 F1-macro:0.85
----------------Task: 1-----------------
Accuracy:0.79 Recall:0.79 F1-micro:0.79
F1_total:1.6401
```

验证集效果并不理想, 未线上提交。


### 新的实验


重新训练一个大模型A：

```
python multi_task_model.py --task=train \
--bert_path=/mnt/sda1/models/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16 \
--epochs=30 \
--batch_size=12 \
--lr=1e-5 \
--data_path=data \
--model_outpath=model_a_wwm_1
--pred_file=data/test.tsv
--pred_outfile=model_a_wwm_1/submit.csv 
```

训练结果：

```
Epoch 28/30
942/942 [==============================] - 348s 369ms/step - loss: 0.0129 - out_0_loss: 0.0029 - out_1_loss: 0.0100 - out_0_acc: 0.9996 - out_1_acc: 0.9978
model score: [(0.8838938053097345, 0.8838938053097345, 0.8773087404213277), (0.7741592920353982, 0.7741592920353982, 0.7741592920353982)] F1_final:1.6515
val_F1: 1.65147, best_val_F1: 1.68861

Epoch 29/30
942/942 [==============================] - 347s 368ms/step - loss: 0.0135 - out_0_loss: 0.0040 - out_1_loss: 0.0095 - out_0_acc: 0.9992 - out_1_acc: 0.9980
model score: [(0.8838938053097345, 0.8838938053097345, 0.878435672923192), (0.7826548672566371, 0.7826548672566371, 0.7826548672566372)] F1_final:1.6611
val_F1: 1.66109, best_val_F1: 1.68861

Epoch 30/30
942/942 [==============================] - 348s 369ms/step - loss: 0.0084 - out_0_loss: 0.0019 - out_1_loss: 0.0065 - out_0_acc: 0.9999 - out_1_acc: 0.9985
model score: [(0.8849557522123894, 0.8849557522123894, 0.8796367767434765), (0.7766371681415929, 0.7766371681415929, 0.7766371681415929)] F1_final:1.6563
val_F1: 1.65627, best_val_F1: 1.68861

正在保存训练数据...
训练曲线图已保存。
正在预测测试集数据...
pred_data: 7596
633/633 [==============================] - 64s 101ms/step
提交文件已生成：./models/submit.csv
正在加载模型...
正在验证数据集...
236/236 [==============================] - 23s 99ms/step
----------------Task: 0-----------------
Accuracy:0.89 Recall:0.89 F1-macro:0.89
----------------Task: 1-----------------
Accuracy:0.80 Recall:0.80 F1-micro:0.80
F1_total:1.6886
```
线上提交得分：1.67085



#### 模型A预测label_j，使用参数：`pred_detail=1`控制输出预测概率：
```
python multi_task_model.py --task=predict \
--bert_path=/mnt/sda1/models/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16 \
--batch_size=12 \
--data_path=data/merge \
--model_outpath=model_a_wwm_1 \
--pred_file=data/label_j.tsv \
--pred_outfile=model_a_wwm_1/pred_label_j.csv \
--pred_detail=1
```

预测结果：
```
729/729 [==============================] - 73s 101ms/step
提交文件已生成：model_a_wwm_1/pred_label_j.csv
```

伪标签数据处理：
```
python data_process.py --task=merge_predict --fname=data/label_j.tsv,model_a_wwm_1/pred_label_j.csv --outpath=data/merge_wwm_1

合并结果已保存到:data/merge_wwm_1\train.tsv
总记录数： 8141
```

####训练模型B，使用新数据继续训练5轮：

```
python multi_task_model.py --task=train \
--bert_path=/mnt/sda1/models/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16 \
--epochs=5 \
--batch_size=12 \
--lr=1e-5 \
--data_path=data/merge_wwm_1 \
--preload_model=model_a_wwm_1/model.weights \
--model_outpath=model_b_wwm_1
```

训练结果：
```
正在预测测试集数据...
pred_data: 7596
633/633 [==============================] - 69s 108ms/step
提交文件已生成：model_b_wwm_1/submit.csv
正在加载模型...
正在验证数据集...
236/236 [==============================] - 25s 106ms/step
----------------Task: 0-----------------
Accuracy:0.84 Recall:0.84 F1-macro:0.82
----------------Task: 1-----------------
Accuracy:0.77 Recall:0.77 F1-micro:0.77
F1_total:1.5917

```

训练效果变差：验证集得分：1.5917， 线上未提交；


#### 训练模型B，使用合并的数据重新训练30轮

把原始数据`data/data/train.tsv` 合并到 `data/merge_wwm_1/train.tsv` 数据后面；
数据集存放到：`data\merge_wwm_1_full`

```
python multi_task_model.py --task=train \
--bert_path=/mnt/sda1/models/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16 \
--epochs=30 \
--batch_size=12 \
--lr=1e-5 \
--data_path=data/merge_wwm_1_full \
--model_outpath=model_b_wwm_1_full
```

训练结果：

```
正在保存训练数据...
训练曲线图已保存。
正在预测测试集数据...
pred_data: 7596
633/633 [==============================] - 61s 97ms/step
提交文件已生成：model_b_wwm_1_full/submit.csv
正在加载模型...
正在验证数据集...
236/236 [==============================] - 22s 94ms/step
----------------Task: 0-----------------
Accuracy:0.89 Recall:0.89 F1-macro:0.88
----------------Task: 1-----------------
Accuracy:0.79 Recall:0.79 F1-micro:0.79
F1_total:1.6682
```
线上得分： 1.66638


-----------------------------------------

## 预训练模型思路


将比赛提供的'spo.txt'拿来跑一个BERT-base的预训练模型; 
预训练模型： `bert_spo`目录下，注意模型只保存了权重，要使用model.load_weight(xx)来加载

然后使用这个预训练模型来训练：

训练模型A：

```
python multi_task_model.py --task=train \
--bert_path=/mnt/sda1/models/bert_spo \
--epochs=30 \
--batch_size=48 \
--model_outpath=model_a_spo
```

-----------------------------------------

##  pipe任务独立思路 

把两个任务分开，使用单独的模型分别进行训练；

数据处理：

```
python data_process.py --task=data_trans_pipe --outpath=data/data_pipe
```
训练数据分别保存到：`data\data_pipe\label_i`和 `data\data_pipe\label_j`

训练模型label_i：
```
python train_class.py --task=train \
--data_path=data/data_pipe/label_i \
--model_outpath=model_pipe_0/label_i \
--epochs=30 \
--batch_size=48 \
--idx=0
```

数据预测
```
python train_class.py --task=predict \
--data_path=data/data_pipe/label_i \
--model_outpath=model_pipe_0/label_i \
--batch_size=48 \
--idx=0
```

```
159/159 [==============================] - 25s 158ms/step
预测结果已保存:model_pipe_0/label_i/submit.csv
```

数据验证
```
python train_class.py --task=eval \
--data_path=data/data_pipe/label_i \
--model_outpath=model_pipe_0/label_i \
--batch_size=48 \
--idx=0
```
线上得分： Accuracy:0.90 Recall:0.90 F1-macro:0.89

命令错误又跑了一遍：

训练结果：

```
Epoch 30/30
429/429 [==============================] - 195s 454ms/step - loss: 0.0258 - accuracy: 0.9936
val_acc: 0.88719, best_val_acc: 0.90293

正在保存训练数据...
训练曲线图已保存。
正在预测数据...
pred_data: 7596
159/159 [==============================] - 23s 147ms/step
预测结果已保存:model_pipe_0/label_i/submit.csv
正在加载模型...
正在验证数据集...
48/48 [==============================] - 7s 148ms/step
预测数据用时7110.984毫秒.
Accuracy:0.90 Recall:0.90 F1-micro:0.90
```



#### 训练模型label_j

训练模型命令：
```
python train_class.py --task=train \
--data_path=data/data_pipe/label_j \
--model_outpath=model_pipe_0/label_j \
--epochs=30 \
--batch_size=48 \
--idx=1
```

训练结果：
```
Epoch 30/30
265/265 [==============================] - 121s 458ms/step - loss: 0.1330 - accuracy: 0.9633
val_acc: 0.77778, best_val_acc: 0.77849

正在保存训练数据...
训练曲线图已保存。
正在预测数据...
pred_data: 7596
159/159 [==============================] - 24s 150ms/step
预测结果已保存:model_pipe_0/label_j/submit.csv
正在加载模型...
正在验证数据集...
30/30 [==============================] - 5s 153ms/step
预测数据用时4581.246毫秒.
Accuracy:0.78 Recall:0.78 F1-micro:0.78
```

把两个结果串起来:
```
python data_process.py --task=merge_pipe --fname=model_pipe_0
```

预测结果文件已合并保存至:model_pipe_0\submit.csv

线上提交结果： 1.66157

### 使用大模型的pipe

训练模型label_i：
```
python train_class.py --task=train \
--bert_path=/mnt/sda1/models/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16 \
--data_path=data/data_pipe/label_i \
--model_outpath=model_pipe_wwm/label_i \
--epochs=30 \
--batch_size=12 \
--idx=0
```

训练结果：
```
Epoch 30/30
1716/1716 [==============================] - 633s 369ms/step - loss: 0.0028 - accuracy: 0.9992
val_acc: 0.90512, best_val_acc: 0.91124

正在保存训练数据...
训练曲线图已保存。
正在预测数据...
pred_data: 7596
633/633 [==============================] - 70s 110ms/step
预测结果已保存:model_pipe_wwm/label_i/submit.csv
正在加载模型...
正在验证数据集...
191/191 [==============================] - 21s 111ms/step
预测数据用时21214.324毫秒.
Accuracy:0.9112 Recall:0.9112 F1-macro:0.9082

```


#### 训练模型label_j

训练模型命令：
```
python train_class.py --task=train \
--bert_path=/mnt/sda1/models/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16 \
--data_path=data/data_pipe/label_j \
--model_outpath=model_pipe_wwm/label_j \
--epochs=30 \
--batch_size=12 \
--idx=1
```
训练结果：

```
Epoch 29/30
1060/1060 [==============================] - 397s 374ms/step - loss: 0.0063 - accuracy: 0.9988
val_acc: 0.78627, best_val_acc: 0.80609

Epoch 30/30
1060/1060 [==============================] - 396s 374ms/step - loss: 0.0054 - accuracy: 0.9991
val_acc: 0.79193, best_val_acc: 0.80609

正在保存训练数据...
训练曲线图已保存。
正在预测数据...
pred_data: 7596
633/633 [==============================] - 64s 101ms/step
预测结果已保存:model_pipe_wwm/label_j/submit.csv
正在加载模型...
正在验证数据集...
118/118 [==============================] - 12s 102ms/step
预测数据用时12088.152毫秒.
Accuracy:0.8061 Recall:0.8061 F1-micro:0.8061
```

合并结果：
```
python data_process.py --task=merge_pipe --fname=model_pipe_wwm
```
预测结果文件已合并保存至:model_pipe_wwm\submit.csv

线上得分： 1.66834


-----------------------------------------

## 串接数据模型

模型结构不变，将输入的字段用 [SEP] 来串接；

```
python data_process.py --task=data_trans_split --outpath=data/data_split
```
训练数据目录：`data/data_split`

模型增加对抗训练；

```
python multi_task_model.py --task=train \
--bert_path=/mnt/sda1/models/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16 \
--epochs=30 \
--batch_size=12 \
--lr=1e-5 \
--data_path=data/data_split \
--model_outpath=model_split_wwm \
--splited=1 \
--adv=1 
```


win下模拟训练命令行:
```
python multi_task_model.py --task=train --epochs=1 --batch_size=16 --lr=1e-5 --data_path=data/data_split --model_outpath=model_split_wwm --splited=1 --adv=1 
```


训练结果：
```
Epoch 30/30
1060/1060 [==============================] - 996s 940ms/step - loss: 0.0385 - out_0_loss: 0.0061 - out_1_loss: 0.0325 - out_0_acc: 0.9993 - out_1_acc: 0.9925
model score: [(0.8903043170559094, 0.8903043170559094, 0.8878684292720402), (0.778485491861288, 0.778485491861288, 0.7784854918612879)] F1_final:1.6664
val_F1: 1.66635, best_val_F1: 1.69013

正在保存训练数据...
训练曲线图已保存。
正在预测测试集数据...
pred_data: 7596
633/633 [==============================] - 64s 102ms/step
提交文件已生成：model_split_wwm/submit.csv
正在加载模型...
正在验证数据集...
118/118 [==============================] - 12s 99ms/step
----------------Task: 0-----------------
Accuracy:0.90 Recall:0.90 F1-macro:0.89
----------------Task: 1-----------------
Accuracy:0.80 Recall:0.80 F1-micro:0.80
F1_total:1.6901

```

-----------------------------------------

## 参考资料


模型训练Tricks——伪标签-半监督学习 - 知乎  https://zhuanlan.zhihu.com/p/365178718

```
伪标签学习（半监督）的几种常用策略：

一、基于标注数据训练模型Model1 
——>基于Model1预测无标注数据 
——> 根据无标注数据的预测概率区分高置信度样本 
——> 筛选高置信样本（伪标签数据）加入标注数据 
——> 训练出新模型Model2。

二、
基于标注数据训练模型Model1 ——>基于Model1预测无标注数据 
——> 根据无标注数据的预测概率区分高置信度样本 
——> 筛选高置信样本（伪标签数据）加入标注数据 
——> 训练新模型Model2 
——> 用Model2替换Model1重复上述步骤，直至模型效果不再提升。

三、可以将标注数据和伪标签数据的损失分配不同的权重。


另一种伪标签的思路：

以文本分类为例，计算测试集中数据与训练集中数据的语义相似度（比如weord2vec，或者更抽象的语义编码表示），将相应高相似度的训练集数据标签，赋给测试集数据。
```



21年 就比过一次了
http://challenge.xfyun.cn/topic/info?type=disease-claims

https://www.bilibili.com/video/BV1hq4y1r7xB?p=15
答辩视频

https://1024.iflytek.com/liveroom
比赛总结：https://mp.weixin.qq.com/s/Wz3Wg62_2SOcXGbod_9yVQ
加入微信竞赛群，请加小助手 coggle666


```
MSE = lambda y_true,y_pred: np.sum(np.power((y_true.reshape(-1,1) - y_pred),2))/len(y_true)

MSE=np.sum(np.power((testY.reshape(-1,1) - predicY),2))/len(testY)

y_true = np.array([2,2,2])
y_pred = np.array([3,1.5,2])
y_pred = np.array([3,1,2])

mse = MSE(y_true, y_pred)

AVE_MSE = lambda x: np.sum(np.power(( np.ones_like(x)*np.average(x) - x),2))/len(x)
AVE_MSE(y_pred)
```

soul-天津-王亚伟(9727464)  17:15:33
https://drive.google.com/file/d/1ccXRvaeox5XCNP_aSk_ttLBY695Erlok/view      医疗预训练模型

@可西哥 可以试试这个， 估计能提1-2个点，


感觉几个方向：
1. -1数据集如何用上；
2. 各字段如何拆分，模型结构的调整；
3. spo信息的利用

