from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer
import torch
import os
import pickle as pkl
import pandas as pd

bert_path = 'roberta_pretrain'
data_path = 'data/data_train.csv'
dataset_pkl = 'data/dataset.pkl'
tokenizer = BertTokenizer.from_pretrained(bert_path)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Data(Dataset):
    def __init__(self, path, train=True):
        self.data = []
        df = pd.read_csv(path)
        if train:
            content = df['content'].to_list()[:20580]
            label_1 = df['label_i'].to_list()[:20580]
            label_2 = df['label_j'].to_list()[:20580]
            for idx, con in tqdm(enumerate(content)):
                encode_dict = tokenizer.encode_plus(con, max_length=280, padding='max_length',
                                                    truncation=True)
                self.data.append((encode_dict['input_ids'], encode_dict['token_type_ids'],
                                  encode_dict['attention_mask'], int(label_1[idx]), int(label_2[idx])))
        else:
            content = df['content'].to_list()[20580:]
            label_1 = df['label_i'].to_list()[20580:]
            label_2 = df['label_j'].to_list()[20580:]
            for idx, con in tqdm(enumerate(content)):
                encode_dict = tokenizer.encode_plus(con, max_length=280, padding='max_length',
                                                    truncation=True)
                self.data.append((encode_dict['input_ids'], encode_dict['token_type_ids'],
                                  encode_dict['attention_mask'], int(label_1[idx]), int(label_2[idx])))

    def __getitem__(self, item):
        self.content = self.data[item]
        return self.content

    def __len__(self):
        return len(self.data)

def get_loader():
    if os.path.exists(dataset_pkl):
        dataset = pkl.load(open(dataset_pkl, 'rb'))
        train_loader = dataset['train']
        # test_data = dataset['test']
        dev_loader = dataset['dev']
    else:
        dataset = {}
        train_data = Data(path=data_path, train=True)
        dev_data = Data(path=data_path, train=False)
        train_loader = data_loader(train_data)
        dev_loader = data_loader(dev_data)
        dataset['train'] = train_loader
        dataset['dev'] = dev_loader
        pkl.dump(dataset, open(dataset_pkl, 'wb'))
    return train_loader,  dev_loader

def collate_fn(batch):
    input_ids, token_type_ids, mask, label_1, label_2 = zip(*batch)
    input_ids = torch.LongTensor(input_ids).to(device)
    token_type_ids = torch.LongTensor(token_type_ids).to(device)
    mask = torch.LongTensor(mask).to(device)
    label_1 = torch.LongTensor(label_1).to(device)
    label_2 = torch.LongTensor(label_2).to(device)

    return (input_ids, mask, token_type_ids), (label_1, label_2)


def data_loader(data):
    loader = DataLoader(data, batch_size=32, shuffle=True, collate_fn=collate_fn)
    return loader


if __name__ == '__main__':
    train_loader, dev_loader = get_loader()
    for content, label in train_loader:
        print(content[0])
        print(label[0])
        print(label[1])
        break