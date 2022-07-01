# 非标准化疾病诉求的简单分诊挑战赛2.0


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

模型结构：

```
Model: "MModel"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
Input-Token (InputLayer)        (None, None)         0
__________________________________________________________________________________________________
Input-Segment (InputLayer)      (None, None)         0
__________________________________________________________________________________________________
BERT_base (Model)               (None, 768)          101677056   Input-Token[0][0]
                                                                 Input-Segment[0][0]
__________________________________________________________________________________________________
AGE-Token (InputLayer)          (None, 1)            0
__________________________________________________________________________________________________
Layer_dense (Dense)             (None, 256)          196864      BERT_base[1][0]
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 257)          0           AGE-Token[0][0]
                                                                 Layer_dense[0][0]
__________________________________________________________________________________________________
out_0 (Dense)                   (None, 20)           5160        concatenate_1[0][0]
__________________________________________________________________________________________________
out_1 (Dense)                   (None, 61)           15738       concatenate_1[0][0]
==================================================================================================
Total params: 101,894,818
Trainable params: 101,894,818
Non-trainable params: 0
__________________________________________________________________________________________________
model input: [<tf.Tensor 'AGE-Token:0' shape=(?, 1) dtype=float32>, <tf.Tensor 'Input-Token:0' shape=(?, ?) dtype=float3
2>, <tf.Tensor 'Input-Segment:0' shape=(?, ?) dtype=float32>]
model output: [<tf.Tensor 'out_0/Softmax:0' shape=(?, 20) dtype=float32>, <tf.Tensor 'out_1/Softmax:0' shape=(?, 61) dty
pe=float32>]
```

训练模型A(使用BERT-base模型)：
```
python multi_task_age.py --task=train --epochs=30 --batch_size=48 --data_path=data/data_age --model_outpath=model_a_age 
--pred_file=model_a_age/test.tsv
--pred_outfile=model_a_age/submit.csv

--preload_model=model_a_age/model.weights 
```
