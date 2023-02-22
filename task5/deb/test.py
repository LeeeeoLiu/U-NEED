import pickle as pkl
from data_clean import dataset, score_model
from torch.utils.data import Dataset, DataLoader
import torch
from scipy.stats import pearsonr, spearmanr
import numpy as np

def raw_data(data_path):
    with open(data_path+'_dia.pkl', 'rb') as f:
        dia = pkl.load(f)

    with open(data_path+'_label.pkl', 'rb') as f:
        label = pkl.load(f)

    #print(len(dia[0][0]),label[0][0])
    return dia,label

def inspect(label):
    n = 0
    for i in label:
        if i < 0.001:
            n += 1
    print(n/len(label),n,len(label))

def test(model, loader):
    device = torch.device("cuda:{}".format('0'))
    model.eval()
    scores = []
    with torch.no_grad():
        for x,y in loader:
            x = {k:torch.squeeze(v,1) for k,v in x.items()}
            x = {k:v.to(device) for k,v in x.items()}
            y = y.to(device)
            _,score = model.forward(x)
            scores.append(score.item())
    return scores

def cor(score, label):
    p = pearsonr(score, label)
    s = spearmanr(score, label)
    print(p,s)

def get_cos_similar(v1: list, v2: list):
    num = float(np.dot(v1, v2))  # 向量点乘
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return num / denom if denom != 0 else 0

if __name__ == '__main__':
    mode = 'phone'
    num_epoch = 5
    dias,labels = raw_data('./dataset/'+mode)
    test_dataset = dataset(dias[2],labels[2])
    test_dataloader = DataLoader(test_dataset, shuffle=False)
    model = score_model()
    '''
    label = labels[0]+labels[1]+labels[2]
    inspect(label)
    

    state_dict = torch.load('./ckpt/recall_1.ckpt',map_location='cuda:0')
    model.load_state_dict(state_dict)
    scores = test(model,test_dataloader)
    cor(scores,labels[2])
    print(get_cos_similar(np.array(scores), np.array(labels[2])))
    '''
    for i in range(num_epoch):
        state_dict = torch.load('./ckpt/phone_{}.ckpt'.format(i),map_location='cuda:0')
        model.load_state_dict(state_dict)
        scores = test(model,test_dataloader)
        cor(scores,labels[2])
        print(get_cos_similar(np.array(scores), np.array(labels[2])))
        