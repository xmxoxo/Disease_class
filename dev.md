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
python multi_task_model.py --task=train --epochs=1 --bert_path=预训练目录 
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

	模型B即为最后得到的模型；


思路二： **分段训练法**

* 第一步：（与 伪标签法 第一步相同）

	对于-1标签的样本，先从数据集中拆分出来，称为“-1样本集”，

	把剩下的完整数据集拿去训练模型，得到模型A；

* 第二步：

	加载模型A权重， 冻结label_j模型层，使用 “-1样本集” 对模型进行继续训练， 得到最终模型B

对比后感觉还是 **伪标签法** 比较好，对label_j再做了一次训练；



