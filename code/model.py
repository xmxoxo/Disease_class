# 卷积核尺寸4*1
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
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


class Model(nn.Module):
    def __init__(self, bert_path):
        super(Model, self).__init__()
        self.config = BertConfig.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path)
        self.linear_1 = nn.Linear(self.config.hidden_size, 20)
        self.linear_2 = nn.Linear(self.config.hidden_size, 61)

    def forward(self, x):
        input_ids = x[0]
        mask = x[1]
        outputs = self.bert(input_ids, attention_mask=mask, token_type_ids=None)
        _, out_pool = outputs[0], outputs[1]  # out.shape:[batch_size,seq_len,hidden_size]
        # out_pool.shape:[32,768]
        out_1 = self.linear_1(out_pool)
        out_2 = self.linear_2(out_pool)
        return out_1, out_2










