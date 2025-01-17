import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from torch.utils.data import Dataset, DataLoader

df = pd.read_excel('default of credit card clients.xls', header=1)
cont_cols = ['LIMIT_BAL','AGE','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6',	'PAY_AMT1',	'PAY_AMT2',	'PAY_AMT3',	'PAY_AMT4',	'PAY_AMT5',	'PAY_AMT6']
cat_cols = ['SEX','EDUCATION','MARRIAGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
target_col = 'default payment next month'

def data_split(df):
    test_size = 0.3
    random_state = 1234
    df_train,df_test = train_test_split(df, test_size=test_size, random_state = random_state, stratify =  df[target_col])
    return df_train,df_test

df_train, df_test = data_split(df)

# Check if all levels in categorical features are present in training set
def check_col(col_name):
  ref = list(df[col_name].unique())
  tar_col = list(df_train[col_name].unique())
  chk = [elem for elem in ref if elem not in tar_col]
  return chk

chk_cols= []
for col in cat_cols:
  temp_lst = check_col(col)
  if len(temp_lst)>0:
    chk_cols.append(col)

# Create a mapping from 0 to n-1 for each level in every categorical feature
cat_code_dict= {}
for col in cat_cols:
  temp = df_train[col].astype('category')
  cat_code_dict[col] = {val:idx for idx,val in enumerate(temp.cat.categories)}

key_name = 3
print(cat_cols[key_name],cat_code_dict[cat_cols[key_name]])

embedding_size_dict = {key: len(val) for key, val in cat_code_dict.items()}


embedding_dim_dict= {key: min(50,val//2) for key,val in embedding_size_dict.items()}


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict
import pickle
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from matplotlib import pyplot as plt

class Process_Dataset:
    def __init__(self,path,cat_cols,cont_cols,target_col):
        self.cat_cols = cat_cols
        self.cont_cols = cont_cols
        self.target_col = target_col
        self.df= pd.read_excel(path, header=1) ##Load the dataset
        self.cat_cols = cat_cols   ##Initialize all categorical features
        self.cont_cols = cont_cols  ##Initialize all continuous features
        self.target_col = target_col ##Set the target feature
        self.data_split()  ##Split the data into train and test set
        self._preprocess()
        self.scaler = StandardScaler()
        self.df_train = self._process(self.df_train,1)
        self.df_test = self._process(self.df_test)
        self.df_val = self._process(self.df_val)

    def data_split(self):
        '''
        Splits the data into 60% train set , 30% val set and 10% test set
        '''
        test_size = 0.1
        val_size = 0.3
        random_state = 1234
        self.df_train,self.df_test = train_test_split(self.df, test_size=test_size, random_state = random_state, stratify = self.df[target_col])
        self.df_train,self.df_val = train_test_split(self.df_train, test_size= val_size, random_state = random_state, stratify = self.df_train[target_col])

    def _preprocess(self):

        '''
         Creates a mapping from 0 to n-1 for each level in every categorical feature
        '''
        self.cat_code_dict= {}
        for col in self.cat_cols:
            temp = self.df_train[col].astype('category')
            self.cat_code_dict[col] = {val:idx for idx,val in enumerate(temp.cat.categories)}


    def _process(self,_df, flag=0):
        '''
        We scale numerical variables using StandardScaler from scikit-learn
        '''
        _df = _df.copy()
        if flag:
            self.scaler.fit(_df[cont_cols])

        # numeric fields
        _df[self.cont_cols] = self.scaler.transform(_df[cont_cols])
        _df[self.cont_cols] = _df[self.cont_cols].astype(np.float32)

        # categorical fields
        for col in self.cat_cols:
            code_dict = self.cat_code_dict[col]
            _df[col] = _df[col].map(code_dict).astype(np.int64)

        # Target
        _df[target_col] = _df[self.target_col].astype(np.float32)
        return _df

class EntityEmbeddingNN(nn.Module):
    def __init__(self,cat_code_dict,cat_cols,cont_cols,target_col,n_classes):
        super().__init__()
        self.cat_code_dict = cat_code_dict
        self.cat_cols = cat_cols   ##Initialize all categorical features
        self.cont_cols = cont_cols  ##Initialize all continuous features
        self.target_col = target_col ##Set the target feature
        self.embeddings = self._create_embedding_vectors()
        self.in_features = self.total_embed_dim + len(cont_cols)

        self.layers = nn.Sequential(
            nn.Linear(self.in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, n_classes)
        )

    def _create_embedding_vectors(self):
        '''
        Create Embedding Layer for each of the categorical variable in dataset
        '''
        ##Get no of levels in each categorical variable and store in dictionary
        self.embedding_size_dict = {key: len(val) for key, val in self.cat_code_dict.items()}
        ##Determine dimension of embeddng vector for each categorical variable
        self.embedding_dim_dict = { key:  min(5, val // 2) for key, val in self.embedding_size_dict.items()}
        embeddings = {}
        self.total_embed_dim = 0
        for col in self.cat_cols:
            num_embeddings = self.embedding_size_dict[col]
            embedding_dim = self.embedding_dim_dict[col]
            embeddings[col] = nn.Embedding(num_embeddings, embedding_dim)
            self.total_embed_dim += embedding_dim
        return nn.ModuleDict(embeddings)

    def forward(self, cat_tensor, num_tensor):
        embedding_tensor_group = []
        for idx, col in enumerate(self.cat_cols):
            layer = self.embeddings[col]
            out = layer(cat_tensor[:, idx])
            embedding_tensor_group.append(out)

            embed_tensor = torch.cat(embedding_tensor_group, dim=1)
            
        out_tensor = torch.cat((embed_tensor, num_tensor), dim=1)
        out_tensor = self.layers(out_tensor)

        return out_tensor


class TabularDataset(Dataset):
    def __init__(self, df, cat_cols,cont_cols,target_col):
        self.cat_cols = cat_cols
        self.cont_cols = cont_cols
        self.target_col = target_col
        self.df = df

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        cat_array = self.df[self.cat_cols].iloc[idx].values
        cont_array = self.df[self.cont_cols].iloc[idx].values
        target_array = self.df[self.target_col].iloc[idx]
        cat_array = torch.LongTensor(cat_array)
        cont_array = torch.FloatTensor(cont_array)

        return cont_array, cat_array, target_array

cont_cols = ['LIMIT_BAL',	'AGE',	'BILL_AMT1',	'BILL_AMT2',	'BILL_AMT3',	'BILL_AMT4',	'BILL_AMT5',	'BILL_AMT6',	'PAY_AMT1',	'PAY_AMT2',	'PAY_AMT3',	'PAY_AMT4',	'PAY_AMT5',	'PAY_AMT6']
cat_cols = ['SEX',	'EDUCATION',	'MARRIAGE',	'PAY_0',	'PAY_2',	'PAY_3',	'PAY_4']
target_col = 'default payment next month'

print(f"We will use {len(cat_cols)} categorical features")
print(f"We will use {len(cont_cols)} continuous features")


###First do all the preprocessing (Scaling and splitting dataset)####
dataset = Process_Dataset("default of credit card clients.xls",cat_cols,cont_cols,target_col)
##Create train and test instances of Dataset class##
dataset_train= TabularDataset(dataset.df_train, cat_cols, cont_cols,target_col)
dataset_test= TabularDataset(dataset.df_test, cat_cols, cont_cols,target_col)
dataset_val= TabularDataset(dataset.df_val, cat_cols, cont_cols,target_col)
##Create train and test dataloaders##
train_loader = DataLoader(dataset_train,batch_size=128, num_workers=32,drop_last=True)
test_loader = DataLoader(dataset_test, batch_size=128, num_workers=32,drop_last=True)
val_loader = DataLoader(dataset_val, batch_size=128, num_workers=32,drop_last=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = EntityEmbeddingNN(dataset.cat_code_dict, cat_cols,cont_cols,target_col,1)
model= model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
train_loss_per_iter = []
train_loss_per_batch = []

test_loss_per_iter = []
test_loss_per_batch = []

n_epochs=16
for epoch in range(n_epochs):
    running_loss = 0.0
    for idx, (cont_array, cat_array, target_array) in enumerate(train_loader):
        cont_array = cont_array.to(device)
        cat_array = cat_array.to(device)
        target_array = target_array.to(device)

        outputs = model(cat_array,cont_array)
        loss = F.binary_cross_entropy_with_logits(outputs.squeeze(1),target_array)
        # Zero the parameter gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_loss_per_iter.append(loss.item())


    train_loss_per_batch.append(running_loss / (idx + 1))
    running_loss = 0.0

    model.eval()
    with torch.no_grad():
        for idx, (cont_array, cat_array, target_array) in enumerate(test_loader):
            cont_array = cont_array.to(device)
            cat_array = cat_array.to(device)
            target_array = target_array.to(device)

            outputs = model(cat_array,cont_array)
            loss = F.binary_cross_entropy_with_logits(outputs.squeeze(1),target_array)

            running_loss += loss.item()
            test_loss_per_iter.append(loss.item())

    test_loss_per_batch.append(running_loss / (idx + 1))
    running_loss = 0.0

def predict(model):
    y_pred=[]
    y_actual=[]
    model.eval()
    with torch.no_grad():
        for idx, (cont_array, cat_array, target_array) in enumerate(val_loader):
            y_actual.append(target_array)
            cont_array = cont_array.to(device)
            cat_array = cat_array.to(device)
            target_array = target_array.to(device)
            outputs = model(cat_array,cont_array)
            y_prob = torch.sigmoid(outputs).cpu().numpy()
            y_pred.append(y_prob)


    y_pred = np.array([elem for ind_list in y_pred for elem in ind_list])
    y_actual = np.array([elem for ind_list in y_actual for elem in ind_list])

    return y_pred,y_actual

def compute_score(y_true, y_pred, round_digits=3):
    log_loss = round(metrics.log_loss(y_true, y_pred), round_digits)
    auc = round(metrics.roc_auc_score(y_true, y_pred), round_digits)

    precision, recall, threshold = metrics.precision_recall_curve(y_true, y_pred)
    f1 = 2 * (precision * recall) / (precision + recall)
    mask = ~np.isnan(f1)
    f1 = f1[mask]
    precision = precision[mask]
    recall = recall[mask]

    best_index = np.argmax(f1)
    threshold = round(threshold[best_index], round_digits)
    precision = round(precision[best_index], round_digits)
    recall = round(recall[best_index], round_digits)
    f1 = round(f1[best_index], round_digits)

    return {
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'threshold': threshold,
        'log_loss': log_loss
    }

y_pred,y_actual = predict(model)

print(compute_score(y_actual,y_pred))