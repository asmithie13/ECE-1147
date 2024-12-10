from PIL import Image
import pandas as pd
import numpy as np
import data_preprocessing
import os

abortion_dev_data: pd.DataFrame
abortion_train_data: pd.DataFrame
gc_dev_data: pd.DataFrame
gc_train_data: pd.DataFrame

abortion_dev_dict: dict
abortion_train_dict: dict
gc_dev_dict: dict
gc_train_dict: dict

[abortion_dev_data, abortion_dev_dict, abortion_train_data, abortion_train_dict,
            gc_dev_data, gc_dev_dict, gc_train_data, gc_train_dict] = data_preprocessing.data_preproc()

features = ['stance', 'persuasiveness']
ind2feat = {ind:feat for ind, feat in enumerate(features)}
feat2ind = {feat:ind for ind, feat in enumerate(features)}

from transformers import BertTokenizer, BertModel, BertConfig
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from sklearn import metrics
import torch

abortion_train_data.insert(6, "list", abortion_train_data[abortion_train_data.columns[3:5]].to_numpy().tolist())
abortion_train_feat = abortion_train_data[["tweet_text", "list"]].copy()
abortion_dev_data.insert(6, "list", abortion_dev_data[abortion_dev_data.columns[3:5]].to_numpy().tolist())
abortion_dev_feat = abortion_dev_data[["tweet_text", "list"]].copy()

class CustomData(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.tweet_text = dataframe["tweet_text"]
        self.targets = self.data["list"]
        self.max_len = max_len
    def __len__(self):
        return len(self.tweet_text)
    def __getitem__(self, ind):
        tweet_text = str(self.tweet_text[ind])
        tweet_text = " ".join(tweet_text.split())
        
        inputs = self.tokenizer.encode_plus(
            tweet_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        
        return {"ids": torch.tensor(ids, dtype=torch.long),
                "mask": torch.tensor(mask, dtype=torch.long),
                "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                "targets": torch.tensor(self.targets[ind], dtype=torch.float)}
MAX_LEN = 200
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-03
train_size = .8
device = 'cpu'
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# training_data = abortion_train_feat.sample(frac=train_size, random_state=200)
# testing_data = abortion_train_feat.drop(training_data.index).reset_index(drop=True)
# training_data = training_data.reset_index(drop=True)
training_data = abortion_train_feat
testing_data = abortion_dev_feat

training_set = CustomData(training_data, tokenizer, MAX_LEN)
testing_set = CustomData(testing_data, tokenizer, MAX_LEN)

train_params = {"batch_size": TRAIN_BATCH_SIZE,
                "shuffle": True,
                "num_workers": 0}
test_params = {"batch_size": VALID_BATCH_SIZE,
                "shuffle": True,
                "num_workers": 0}

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 2)
    
    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output

model = BERTClass()
model.to(device)

def loss_fxn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

def train(epoch):
    model.train()
    for _,data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fxn(outputs, targets)
        if _%5000==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

for epoch in range(EPOCHS):
    train(epoch)
    
def validation(epoch):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets

for epoch in range(EPOCHS):
    outputs, targets = validation(epoch)
    outputs = np.array(outputs) >= 0.5
    accuracy = metrics.accuracy_score(targets, outputs)
    f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
    f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")

gc_train_data.insert(6, "list", gc_train_data[gc_train_data.columns[3:5]].to_numpy().tolist())
gc_train_feat = gc_train_data[["tweet_text", "list"]].copy()
gc_dev_data.insert(6, "list", gc_dev_data[gc_dev_data.columns[3:5]].to_numpy().tolist())
gc_dev_feat = gc_dev_data[["tweet_text", "list"]].copy()
    
training_data = gc_train_feat
testing_data = gc_dev_feat

training_set = CustomData(training_data, tokenizer, MAX_LEN)
testing_set = CustomData(testing_data, tokenizer, MAX_LEN)

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

model = BERTClass()
model.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    train(epoch)

for epoch in range(EPOCHS):
    outputs, targets = validation(epoch)
    outputs = np.array(outputs) >= 0.5
    accuracy = metrics.accuracy_score(targets, outputs)
    f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
    f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")
