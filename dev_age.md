# 非标准化疾病诉求的简单分诊挑战赛2.0


## 新思路：多维度融合

把年龄这个特征单独拿出来，与文本得到的特征进行融合；

```
年龄 ==========>━┓
                  ┣━》全连接层==》输出
文本 ==>BERT ==>━┛
```

###　模型结构

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

### 数据预处理


模型预处理命令：
```
python data_process.py --task=data_trans_age
```

数据生成到目录:`./data_age`目录下；

同时生成label_j的训练数据：`./data_age/label_j`


### 模型训练 

训练模型A(使用BERT-base模型, 带降维到256)：
```
python multi_task_age.py --task=train \
--epochs=30 \
--batch_size=48 \
--data_path=data_age \
--model_outpath=model_a_age \
--pred_file=data_age/test.tsv \
--pred_outfile=model_a_age/submit.csv
```

模型验证与预测：

```
python multi_task_age.py --task=eval \
--data_path=data_age \
--model_outpath=model_a_age
```

```
python multi_task_age.py --task=predict \
--data_path=data_age \
--model_outpath=model_a_age \
--pred_file=data_age/test.tsv \
--pred_outfile=model_a_age/submit.csv
```

训练结果：

```
----------------Task: 0-----------------
Accuracy:0.87 Recall:0.87 F1-macro:0.87
----------------Task: 1-----------------
Accuracy:0.76 Recall:0.76 F1-micro:0.76
F1_total:1.6264
```


重新训练模型A(使用BERT-base模型, 不带降维层)：

```
python multi_task_age.py --task=train \
--epochs=30 \
--batch_size=48 \
--data_path=data_age \
--model_outpath=model_a_age_full \
--pred_file=data_age/test.tsv \
--pred_outfile=model_a_age_full/submit.csv
```

训练结果：
```
Epoch 28/30
236/236 [==============================] - 105s 446ms/step - loss: 0.2363 - out_0_loss: 0.0478 - out_1_loss: 0.1885 - out_0_acc: 0.9898 - out_1_acc: 0.9479
model score: [(0.862349610757254, 0.862349610757254, 0.8597250913819829), (0.7572540693559802, 0.7572540693559802, 0.7572540693559802)] F1_final:1.6170
val_F1: 1.61698, best_val_F1: 1.61966

Epoch 29/30
236/236 [==============================] - 105s 444ms/step - loss: 0.2346 - out_0_loss: 0.0456 - out_1_loss: 0.1891 - out_0_acc: 0.9907 - out_1_acc: 0.9474
model score: [(0.8570417551309271, 0.8570417551309271, 0.8550398378976942), (0.7569002123142251, 0.7569002123142251, 0.7569002123142251)] F1_final:1.6119
val_F1: 1.61194, best_val_F1: 1.61966

Epoch 30/30
236/236 [==============================] - 105s 447ms/step - loss: 0.2198 - out_0_loss: 0.0423 - out_1_loss: 0.1774 - out_0_acc: 0.9913 - out_1_acc: 0.9528
model score: [(0.859518754423213, 0.859518754423213, 0.8585317249036144), (0.7607926397735315, 0.7607926397735315, 0.7607926397735314)] F1_final:1.6193
val_F1: 1.61932, best_val_F1: 1.61966

正在保存训练数据...
训练曲线图已保存。
正在预测测试集数据...
pred_data: 7597
159/159 [==============================] - 23s 147ms/step
提交文件已生成：model_a_age_full/submit.csv
正在加载模型...
正在验证数据集...
59/59 [==============================] - 9s 149ms/step
----------------Task: 0-----------------
Accuracy:0.86 Recall:0.86 F1-macro:0.86
----------------Task: 1-----------------
Accuracy:0.76 Recall:0.76 F1-micro:0.76
F1_total:1.6197
```

结果不是很好，未提交线上。


训练模型B(使用BERT-base模型, 不带降维层)：
```
python multi_task_age.py --task=train \
--epochs=10 \
--batch_size=48 \
--data_path=data_age/label_j \
--model_outpath=model_b_age_full \
--preload_model=model_a_age_full/model.weights \
--pred_file=data_age/test.tsv \
--pred_outfile=model_b_age_full/submit.csv \
--frozen=1 
```

