import json as json
import numpy as np
from scipy import stats
from collections import OrderedDict
import pickle as pkl

dialog_tpath = 'dataset/air_train_dataset_0210_encrypt.json'
dialog_dpath = 'dataset/air_dev_dataset_0210_encrypt.json'
dialog_ypath = 'dataset/air_test_dataset_0210_encrypt.json'
action_path = 'dataset/user_action_on_item_air_dataset_with_encrypt_0210.txt'
kg_path = 'dataset/kg_item_air_dataset_with_encrypt_0210.txt'

errs = ['12932974636ed7c205a74e7308a6750e', '129cea052a3cb76086249e267870954e',
'12a645ba667650f2d6c6187024be2aed', '12b0df0a8d9a353f6f4cc1e9fa753cb8',
'2c634b9281a981929f77331e6c243a98', '2d2aba0d700abb8c99004af42969ee88',
'2d845fbeb46cd32875d949fa3d77eeb6', '2d8f78a9b8d385e96b2e0c16cd3c7e78',
'2dda6ae8087115c83d5fccda33192e48','2e50e4723f6f00a304b627c43125d2fa',
'0494856deff96eab20ccfbe6e9256787',
 '17cd8cef3c974bfb92e331c8e6fae409',
 'a1ec184d6c7b50cd4b2391010068e544',
 '1d64d8a4f0359a07cb7d6feddbf5d2ef']

def text(dialog_path):
    with open(dialog_path, 'r', encoding='utf-8') as f:
        whole = json.load(f)

    dialogs = []
    t = []
    for i in whole:
        dialog = []
        turn = ''
        if i['sid'] not in errs:
            s = 0 if (i['dialogue'][0]['sender_type'] == '用户') else 1
            t.append(i)
        else:
            continue
        for n,j in enumerate(i['dialogue']):
            cur_stance = 0 if j['sender_type'] == '用户' else 1
            if cur_stance == s or n == 0:
                turn += j['send_content'] #待增加连接处没有标点的处理
            else:
                dialog.append(turn)
                turn = j['send_content']
            if n == len(i['dialogue'])-1:
                dialog.append(turn)

            s = cur_stance

        dialogs.append(dialog)

    return dialogs,t

def action(action_path):
    actions = []
    with open(action_path, 'r', encoding='utf-8') as f:
        a = f.readline()
        while a:
            a = f.readline()
            actions.append(a.strip().split(','))

    return actions

def traditional(actions,t,dialogs):
    act_dic = {}
    for i in actions[:-1]:
        try:
            time = i[3].split()[0]
            time = ''.join(time.split('-'))
            if i[0]+i[1]+time not in act_dic:
                act_dic[i[0]+i[1]+time] = [i[-1]]
            else:
                act_dic[i[0]+i[1]+time].append(i[-1])
        except: print(i)

    labels = []
    miss = []
    for n,i in enumerate(t):
        if i['sellerid']+i['userid']+i['ds'] in act_dic:
            act = act_dic[i['sellerid']+i['userid']+i['ds']]
        else:
            miss.append(n)
            continue
        if '1' in act:
            labels.append(2)
        elif ('2' in act) or ('3' in act):
            labels.append(1)
        else:
            labels.append(0)

    dia = []
    for n,i in enumerate(dialogs):
        if n in miss:
            continue
        dia.append(i)

    return dia, labels

def tradi_label(paths):
    dias = []
    labels = []
    actions = action(action_path)
    for i in paths:
        dialogs,t = text(i)
        dia,label = traditional(actions, t, dialogs)
        dias.append(dia)
        labels.append(label)

    return dias, labels

def vec(actions):
    kg_list = []
    with open(kg_path, 'r', encoding='utf-8') as f:
        l = f.readline()
        while l:
            l = f.readline()
            kg_list.append(l.strip().split(','))

    kgl = kg_list[:-1]
    items = set()
    for i in kgl:
        items.add(i[0])
    for j in actions[:-1]:
        items.add(j[2])

    item = list(items)
    k_dict = OrderedDict({i:n for n,i in enumerate(item)})
    return k_dict

def create_acd(actions):
    ad = {}
    for n,j in enumerate(actions[:-1]):
        time = j[3].split()[0]
        time = ''.join(time.split('-'))
        if j[0]+j[1]+time in ad:
            ad[j[0]+j[1]+time].append(actions[n])
        else:
            ad[j[0]+j[1]+time] = [actions[n]]

    return ad

def act_vec(t,actions,k_dict):
    miss = []
    be = []
    af = []
    rec_items = []
    act_dict = create_acd(actions)
    for n,i in enumerate(t):
        dia_t = i['dialogue'][0]['time'].split()[1]
        dia_t = ''.join(dia_t.split(':'))[:6]
        be_item = []
        af_item = []
        rec_item = [turn['rec_item_id']  for turn in  i['dialogue'] if turn['rec_item_id']]
        if i['sellerid']+i['userid']+i['ds'] in act_dict:
            idx = i['sellerid']+i['userid']+i['ds']
            for j in act_dict[idx]:
                sec = j[3].split()[1]
                sec = ''.join(sec.split(':'))[:6]
                if int(sec) < int(dia_t):
                    be_item.append([j[2],j[-1]])
                else:
                    af_item.append([j[2],j[-1]])

        if (not be_item) or (not af_item):
            miss.append(n)
        else:
            be.append(be_item)
            af.append(af_item)
            rec_items.append(rec_item)

    return be,af,miss, rec_items


def innovation(dialogs, be, af, miss, rec_items, con_type, k_dict, t):

    dia = []
    whole_js = []
    for n, i, j in zip(range(len(dialogs)), dialogs, t):
        if n in miss:
            continue
        else:
            dia.append(dialogs[n])
            whole_js.append(j)

    b = np.zeros_like(np.array(list(k_dict.keys())),dtype=int)
    a = np.zeros_like(np.array(list(k_dict.keys())),dtype=int)
    # 前后行为差异标签， 0， 1,  2
    label = []
    # 推荐商品与对话后行为重叠标签，离散值，区间为[0,1]
    label_2 = []
    #print(b)
    for i, j, rec_item in zip(be, af, rec_items):
        # j = [ unique_j.split('//') for unique_j in set([f'{j_item}//{j_behavior}' for [j_item, j_behavior] in o_j])]
        # i = [ unique_i.split('//') for unique_i in set([f'{i_item}//{i_behavior}' for [i_item, i_behavior] in o_i])]

        behaviors_before = {}
        for [item, action] in i:
            if action == '0':
                item_weight = 1
            elif action == '1':
                item_weight = 10
            elif action == '2':
                item_weight = 5
            elif action == '3':
                item_weight = 2

            if item in behaviors_before:
                behaviors_before[item] += item_weight
            else:
                behaviors_before[item] = item_weight

        behaviors_after = {}
        for [item, action] in j:
            if action == '0':
                item_weight = 1
            elif action == '1':
                item_weight = 10
            elif action == '2':
                item_weight = 5
            elif action == '3':
                item_weight = 2

            if item in behaviors_after:
                behaviors_after[item] += item_weight
            else:
                behaviors_after[item] = item_weight

        flat_rec_item = [item for sublist in rec_item for item in sublist]
        label_2.append(len(set(flat_rec_item) & set(behaviors_after.keys()))/len(set(flat_rec_item)))

        sorted_behaviors_before = sorted(behaviors_before.items(), key=lambda x: x[1], reverse=True)
        sorted_behaviors_after = sorted(behaviors_after.items(), key=lambda x: x[1], reverse=True)

        union_items = set([item for item, item_weight in sorted_behaviors_before]) | set([item for item, item_weight in sorted_behaviors_after])

        before_item_list = []
        after_item_list = []

        for item in union_items:
            if item in behaviors_before:
                before_item_list.append(behaviors_before[item])
            else:
                before_item_list.append(0)

            if item in behaviors_after:
                after_item_list.append(behaviors_after[item])
            else:
                after_item_list.append(0)

        levene_stat,levene_pvalue = stats.levene(before_item_list,after_item_list)
        if levene_pvalue < 0.05:
            _, p_v = stats.ttest_ind(before_item_list,after_item_list,equal_var=False)
        else:
            _, p_v = stats.ttest_ind(before_item_list,after_item_list)


        # for n in i:
        #     if n[1] > '0':
        #         try:
        #             b[k_dict[n[0]]] += 1
        #         except:
        #             print(k_dict[n[0]])

        # for m in j:
        #     if m[1] > '0':
        #         a[k_dict[m[0]]] += 1

        # levene_stat,levene_pvalue = stats.levene(b,a)
        # if levene_pvalue < 0.05:
        #     _, p_v = stats.ttest_ind(b,a,equal_var=False)
        # else:
        #     _, p_v = stats.ttest_ind(b,a)

        if p_v < 0.01:
            label.append(2)
        elif p_v < 0.05 and p_v >= 0.01:
            label.append(1)
        else:
            label.append(0)

        # label.append(p_v)

    return dia,label, label_2, whole_js

def innv_label(paths, con_type='add', area='美妆行业'):
    dias = []
    labels = []
    labels_2 = []
    actions = action(action_path)
    for i in paths:
        dialogs,t = text(i)
        k_dict = vec(actions)
        be,af,miss, rec_items = act_vec(t, actions, k_dict)
        '''
        print(len(be),len(af),len(miss))
        print(be[0])
        print(af[0])
        '''

        dia, label, label_2, whole_json = innovation(dialogs, be, af, miss, rec_items, con_type,
                                k_dict, t)

        dia, label, label_2 = pick(dia, label, label_2, whole_json, area)

        dias.append(dia)
        labels.append(label)
        labels_2.append(label_2)
    return dias,labels, labels_2

def pick(dia, label, label_2, whole_json, area):
    d = []
    l1 = []
    l2 = []
    for i,j,m,n in zip(dia, label, label_2, whole_json):
        if n['domain'] == area:
            d.append(i)
            l1.append(j)
            l2.append(m)
    return d,l1,l2

def print_interval(labels_2):
    intervals = {
        '0.0-0.2': 0,
        '0.2-0.4': 0,
        '0.4-0.6': 0,
        '0.6-0.8': 0,
        '0.8-1.0': 0
    }
    for _lb2 in labels_2:
        if _lb2 <= 0.2:
            intervals['0.0-0.2'] += 1
        elif _lb2 <= 0.4:
            intervals['0.2-0.4'] += 1
        elif _lb2 <= 0.6:
            intervals['0.4-0.6'] += 1
        elif _lb2 <= 0.8:
            intervals['0.6-0.8'] += 1
        else:
            intervals['0.8-1.0'] += 1

    print('0.0-0.2:\t', intervals['0.0-0.2'], '\t', intervals['0.0-0.2'] / len(labels_2))
    print('0.2-0.4:\t', intervals['0.2-0.4'], '\t', intervals['0.2-0.4'] / len(labels_2))
    print('0.4-0.6:\t', intervals['0.4-0.6'], '\t', intervals['0.4-0.6'] / len(labels_2))
    print('0.6-0.8:\t', intervals['0.6-0.8'], '\t', intervals['0.6-0.8'] / len(labels_2))
    print('0.8-1.0:\t', intervals['0.8-1.0'], '\t', intervals['0.8-1.0'] / len(labels_2))

if __name__ == '__main__':
    label_mode = 'innovation'
    paths = (dialog_tpath,dialog_dpath,dialog_ypath)
    if label_mode == 'innovation':
        dias, labels, labels_2 = innv_label(paths, area='手机行业')
    else:
        dias, labels = tradi_label(paths)
    
    with open('phone_dia.pkl', 'wb') as f:
        pkl.dump(dias, f)
    with open('phone_label.pkl', 'wb') as f:
        pkl.dump(labels_2,f)
    
    print(len(dias[0]), len(labels[0]), len(labels_2[0]))
    print(len(dias[1]), len(labels[1]), len(labels_2[1]))
    print(len(dias[2]), len(labels[2]), len(labels_2[2]))


    print('标签1：用户对话前后行为差异，离散： 0， 1， 2')
    print(labels[0].count(0),'\t',labels[0].count(1),'\t' , labels[0].count(2), '\t', (labels[0].count(1) + labels[0].count(2)) / len(labels[0]))
    print(labels[1].count(0),'\t',labels[1].count(1),'\t' , labels[1].count(2), '\t', (labels[1].count(1) + labels[1].count(2)) / len(labels[1]))
    print(labels[2].count(0),'\t',labels[2].count(1),'\t' , labels[2].count(2), '\t', (labels[2].count(1) + labels[2].count(2)) / len(labels[2]))
    

    print('标签2：推荐与对话后行为重叠比例，连续，区间[0, 1]')
    print('Train:')
    print_interval(labels_2[0])
    print('Valid:')
    print_interval(labels_2[1])
    print('Test:')
    print_interval(labels_2[2])
            

#be = [[['123456','0'],['123','1']],[],...]