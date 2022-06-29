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
python data_process.py
```

生成的数据会保存到`data` 目录下；


## 模型训练

bert4keras架构的多任务模型

```
python multi_task_model.py --task=train --epochs=1 --bert_path=预训练目录 
```

预测数据生成提交文件：

```
python multi_task_model.py --task=eval
```
