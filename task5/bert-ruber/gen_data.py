import pickle as pkl
import json
import re

def raw_data(data_path):
    with open(data_path+'_dia.pkl', 'rb') as f:
        dia = pkl.load(f)

    with open(data_path+'_label.pkl', 'rb') as f:
        label = pkl.load(f)

    #print(len(dia[0][0]),label[0][0])
    return dia,label

def create_txt(dia, label, mode):
    mode = mode+'.txt'
    src_f = open('./data/phone/src-'+mode, 'w', encoding='utf-8')
    tgt_f = open('./data/phone/tgt-'+mode, 'w', encoding='utf-8')
    label_f = open('./data/phone/label-'+mode, 'w', encoding='utf-8')
    for i,j in zip(dia,label):
        i = [turn.strip() for turn in i]
        dic = {}
        dic['id'] = n
        context = ' [SEP] '.join(i[:-1])
        tgt = i[-1].strip()
        tgt = re.sub('\n','',tgt)
        '''
        y = re.search('\\r',tgt)
        if y:
            print(i)
            '''
        src_f.write(context+'\n')
        tgt_f.write(tgt+'\n')
        label_f.write(str(j)+'\n')

if __name__ == '__main__':
    mode = 'phone'
    dias,labels = raw_data('../../purchase/dataset/'+mode)
    
    for dia,label,n in zip(dias,labels,range(3)):
        if n == 0:
            create_txt(dia,label,'train')
        elif n == 1:
            create_txt(dia,label,'dev')
        else:
            create_txt(dia,label, 'test')
            '''
    with open('./data/src-train.txt', 'r', encoding='utf-8') as f:
        con = []
        turn = f.readline()
        #con.append(turn.strip())
        while turn:
            con.append(turn.strip())
            turn = f.readline()
    with open('./data/tgt-train.txt', 'r', encoding='utf-8') as f:
        tgt = []
        turn = f.readline()
        #tgt.append(turn.strip())
        while turn:
            tgt.append(turn.strip())
            turn = f.readline()

    t = '你好\nhaha'
    b = re.sub('\n','',t)
    print(b)

    for i,n in zip(dias[0],range(len(dias[0]))):
        context = con[n].split(' [SEP] ')
        d = context+[tgt[n]]
        i = [turn.strip() for turn in i]
        if i != d:
            print(i,n)
            print(d)

            
    print(len(dias[0]))
    print(dias[0][-1])
    print(len(con),len(tgt))  
    #print(con[0],tgt[0])
    print(con[5516],tgt[5516])
            
'''