import json as json
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
#from sklearn.model_selection import train_test_split
import pickle as pkl
from tqdm import tqdm

def raw_data(data_path):
    with open(data_path+'_dia.pkl', 'rb') as f:
        dia = pkl.load(f)

    with open(data_path+'_label.pkl', 'rb') as f:
        label = pkl.load(f)

    #print(len(dia[0][0]),label[0][0])
    return dia,label

class dataset(Dataset):
    def __init__(self,dia,labels):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
        self.dialogue = ['[SEP]'.join(i) for i in dia]
        self.labels = torch.Tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,idx):
        batch = self.dialogue[idx]
        emb = self.tokenizer(batch,padding='max_length', truncation=True, return_tensors="pt")
        label = self.labels[idx]

        return emb,label

class score_model(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = AutoModel.from_pretrained('bert-base-chinese')
        bert_hidden_size = self.model.config.hidden_size
        mlp_hidden_size_1 = int(bert_hidden_size / 2)
        mlp_hidden_size_2 = int(mlp_hidden_size_1 / 2)
        self.mlp = nn.Sequential(
            nn.Linear(bert_hidden_size, mlp_hidden_size_1),
            nn.ELU(),
            nn.Linear(mlp_hidden_size_1, mlp_hidden_size_2),
            nn.ELU(),
            nn.Linear(mlp_hidden_size_2, 1),
            nn.Sigmoid())
        
        self.device = torch.device("cuda:{}".format('0'))
        self.to(self.device)

    def forward(self, x):
        #print(input_ids.size())
        '''
        input_ids = torch.squeeze(input_ids,0)
        token_type_ids = torch.squeeze(token_type_ids,0)
        attention_mask = torch.squeeze(attention_mask,0)
        '''
        #print(x['input_ids'].size())
        output_dict = self.model(
            **x,
            return_dict=True)
        pooled_output = output_dict['pooler_output']
        score = self.mlp(pooled_output)
        return output_dict, score

    @torch.no_grad()
    def get_score(self, sample: dict):
        self.eval()
        input_ids, token_type_ids, attention_mask = self.encode_ctx_res_pair(
            sample['context'], sample['hyp_response'])
        _, score = self.forward(input_ids, token_type_ids, attention_mask)
        return score[0].item()
        
class trainer(object):
    def __init__(self,model,train,val):
        super().__init__()
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(),eps=1e-6,lr=2e-5)
        self.loss_fn = nn.MSELoss()
        self.trainset = train
        self.val = val
        self.num_epoch = 5
        self.device = torch.device("cuda:{}".format('0'))
        
    def train(self):
        self.model.train()
        for epoch in range(self.num_epoch):
            self.loss_record = []
            train_pbar = tqdm(self.trainset, position=0, leave=True)
            for x,y in train_pbar:
                self.optimizer.zero_grad()
                '''
                x['input_ids'] = x['input_ids'].to(self.device)
                x['token_type_ids'] = x['token_type_ids'].to(self.device)
                x['attention_mask'] = x['attention_mask'].to(self.device)
                '''
                x = {k:torch.squeeze(v,1) for k,v in x.items()}
                x = {k:v.to(self.device) for k,v in x.items()}
                y = y.to(self.device)
                _,score = self.model.forward(x)
                loss = self.loss_fn(score,y)
                loss.backward()
                self.optimizer.step()
                self.loss_record.append(loss.detach().item())

                train_pbar.set_description(f'Epoch [{epoch+1}/{self.num_epoch}]')
                train_pbar.set_postfix({'loss': loss.detach().item()})
            
            mean_train_loss = sum(self.loss_record)/len(self.loss_record)
            print(f'Epoch [{epoch+1}/{self.num_epoch}]: Train loss: {mean_train_loss:.4f}')
            self.val_c()
            torch.save(self.model.state_dict(), 'ckpt/phone_{}.ckpt'.format(epoch))

    def val_c(self):
        self.model.eval()
        loss_record = []
        for x,y in self.val:
            with torch.no_grad():
                x = {k:torch.squeeze(v,1) for k,v in x.items()}
                x = {k:v.to(self.device) for k,v in x.items()}
                y = y.to(self.device)
                _,score = self.model.forward(x)
                loss = self.loss_fn(score,y)
                loss_record.append(loss.detach().item())

        mean_valid_loss = sum(loss_record)/len(loss_record)
        print(f'Valid loss: {mean_valid_loss:.4f}')
        

if __name__ == '__main__':
    mode = 'phone'
    dias,labels = raw_data('./dataset/'+mode)
    train_dataset = dataset(dias[0],labels[0])
    val_dataset = dataset(dias[1],labels[1])
    #test_dataset = dataset(dias[2],labels[2])

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, shuffle=False)
    #test_dataloader = DataLoader(test_dataset, shuffle=False)
    
    model = score_model()
    state_dict = torch.load('./ckpt/recall_1.ckpt',map_location='cuda:0')
    model.load_state_dict(state_dict)
    trainer = trainer(model,train_dataloader,val_dataloader)
    trainer.train()
    #print(train_dataset[0])
    #print(dias[0][0],labels[0][0])