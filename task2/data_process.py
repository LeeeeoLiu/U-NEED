import json
import random
import copy
import numpy as np
from collections import defaultdict
import os
from utils.utils import attr_seq
import matplotlib.pyplot as plt

data_path = "data/finall/air_train_dataset_0210_encrypt.json"
train_path = "data/finall/air_train_dataset_0210_encrypt.json"
dev_path = "data/finall/air_dev_dataset_0210_encrypt.json"
test_path = "data/finall/air_test_dataset_0210_encrypt.json"
ch2en = {'美妆行业': "beauty", '手机行业': "cellphone", '服装行业': "clothes", '鞋类行业': "shoes",'大家电行业': "electronics","all":"all"}
en2ch = {v:k for k,v in ch2en.items()}

def get__new_attrs():
    all_attr = defaultdict(set)
    def get_data_attrs(path):
        data = load_json(path)
        for d in data:
            for m in d["dialogue"]:
                for ar in m["attributes"]:
                    if "key" in ar:
                        all_attr[d["domain"]].add(ar["key"].strip())
        
    get_data_attrs(train_path)
    get_data_attrs(dev_path)
    get_data_attrs(test_path)
    for d in all_attr:
        all_attr[d]=list(all_attr[d])
    all_attr_set = set()
    count = 0
    save_json(list(all_attr.keys()),"data/all_domain.json")
    for d,attr in all_attr.items():
        save_json(attr,f"data/{d}/attrs.json")
        all_attr_set |= set(attr)
        count+=len(attr)
        print(f"领域：{d}，属性数目：{len(attr)}")
    print(f"去重前属性总数：{count}")
    print(f"属性总数：{len(all_attr_set)}")
    all_attr["all"] = len(all_attr_set)
    save_json(all_attr,"data/attrs.json")

def get__attrs():
    path = data_path
    data = load_json(path)
    for d in data:  
        for m in d["dialogue"]:
            for ar in m["attributes"]:
                if "key" in ar:
                    all_attr[d["domain"]].add(ar["key"])
        
    for d in all_attr:
        all_attr[d]=list(all_attr[d])
    all_attr = defaultdict(set)
    all_attr_set = set()
    count = 0
    save_json(list(all_attr.keys()),"data/all_domain.json")
    for d,attr in all_attr.items():
        save_json(attr,f"data/{d}/attrs.json")
        all_attr_set |= set(attr)
        count+=len(attr)
        print(f"领域：{d}，属性数目：{len(attr)}")
    print(f"去重前属性总数：{count}")
    print(f"属性总数：{len(all_attr_set)}")
    all_attr["all"] = len(all_attr_set)
    save_json(all_attr,"data/attrs.json")


def preprocess():
    path = "data/air_dataset_0121_encrypt.json"
    data = json.load(open(path,'r',encoding='utf-8'))
    train_data, dev_data, test_data = split_data(data)
    all_attr = load_json("data/attrs.json")
    all_attr_set = set()
    count = 0
    save_json(list(all_attr.keys()),"data/all_domain.json")
    for d,attr in all_attr.items():
        save_json(attr,f"data/{d}/attrs.json")
        all_attr_set |= set(attr)
        count+=len(attr)
        print(f"领域：{d}，属性数目：{len(attr)}")
    print(f"去重前属性总数：{count}")
    print(f"属性总数：{len(all_attr_set)}")
    all_attr["all"] = len(all_attr_set)
    save_json(all_attr,"data/attrs.json")
    # attr2idx = {a:idx for idx,a in enumerate(all_attr_set)}
    # label_dim = len(attr2idx)
    # def handle_data(data,part):
    #     part_data = []
    #     for d in data:
    #         item = {"domain":d["domain"],"sid":d["sid"],"sellerid":d["sellerid"],"userid":d["userid"]
    #                 ,"context":[],"attr_path":[],"text":"","label":"","domain":d["domain"]}
    #         for m in d["dialogue"]:
    #             attr_dict = {}
    #             for ar in m["attributes"]:
    #                 if "key" in ar:
    #                     attr_dict[ar["key"]] = ar["value"]
    #             # try:
    #             if m["sender_type"]=="客服" and m["act_tag"]=="系统提问":
    #                 item["raw_label"] = attr_dict
    #                 if len(item["raw_label"])!=0:
    #                     label_vec = np.zeros(label_dim)
    #                     for l in item["raw_label"].keys():
    #                         if l in attr2idx:
    #                             label_vec[attr2idx[l]] = 1
    #                     item["label"] = ",".join(map(str,label_vec))
    #                     part_data.append(copy.deepcopy(item))
    #             item["context"].append(m["send_content"])
    #             if len(item["text"])!=0:
    #                 item["text"]+=" "
    #             if  not isinstance(m["send_content"],int):
    #                 item["text"]+= "sos_c "+ m["send_content"] + " eos_c"
    #             attr = attr_dict
    #             item["attr_path"].append(attr)
    #             attr_k = list(attr.keys())
    #             if len(attr_k)>0:
    #                 attr_idx = []
    #                 for a in attr_k:
    #                     attr_idx.append(str(attr2idx[a]))
    #                 item["text"]+= " sos_a "+ " ".join(attr_idx) + " eos_a"
    #             # except:
    #             #     print(m["attributes"],d["sid"])
    #     save_json(part_data, f"data/{part}.json")
    #     print(f"{part}数据量为：{len(part_data)}")
    #     return part_data
    # handle_data(train_data,"train")
    # handle_data(dev_data,"dev")
    # handle_data(test_data,"test")
    
def split_data(data,split_rate=0.05):
    random.shuffle(data)
    train_data = []
    dev_data = []
    test_data = []
    num = int(len(data)*split_rate)
    train_data = data[2*num:]
    dev_data=data[:num]
    test_data=data[num:2*num]
    return train_data, dev_data, test_data

def handle_data(data,part,attr2idx,save=True):
    part_data = []
    label_dim = len(attr2idx)
    statics = defaultdict(int)
    for d in data:
        item = {"domain":d["domain"],"sid":d["sid"],"sellerid":d["sellerid"],"userid":d["userid"]
                ,"context":[],"attr_path":[],"text":"","label":"","domain":d["domain"],"attr_seq":[]}
        for m in d["dialogue"]:
            attr_dict = {}
            for ar in m["attributes"]:
                if "key" in ar:
                    attr_dict[ar["key"].strip()] = ar["value"]
            # try:
            if m["sender_type"]=="客服" and m["act_tag"]=="系统提问":
                item["raw_label"] = attr_dict
                if len(item["raw_label"])!=0:
                    label_vec = np.zeros(label_dim)
                    for l in item["raw_label"].keys():
                        if l in attr2idx:
                            label_vec[attr2idx[l]] = 1
                    statics[str(len(item["raw_label"]))]+=1
                    item["label"] = ",".join(map(str,label_vec))
                    item["attr_seq"] = attr_seq(attr_dict.keys(),attr2idx)
                    item["reply"] = m["send_content"]
                    part_data.append(copy.deepcopy(item))
            item["context"].append(m["send_content"])
            # if  not isinstance(m["send_content"],int):
            #     item["text"]+= "sos_c "+ m["send_content"] + " eos_c"
            attr = attr_dict
            if len(attr)>0:
                item["attr_path"].append(attr)
            attr_k = list(attr.keys())
            item["text"]=" ".join(item["context"])
            # 处理attr_path
            attr_path = []
            for att in item["attr_path"]:
                attr_path.extend(list(att.keys()))
            if len(attr_path)>0:
                attr_idx = []
                for a in attr_path:
                    attr_idx.append(str(attr2idx[a]))
                # item["text"]+= " sos_a "+ " ".join(attr_idx) + " eos_a"
                curr_attr = " ".join(attr_idx)
                if len(item["text"])!=0 and curr_attr:
                    item["text"]+="[SEP]"
                item["text"]+=curr_attr
            # except:
            #     print(m["attributes"],d["sid"])
    if save:
        save_json(part_data, "data/{}/{}.json".format(d["domain"],part))
        print("{} {} 数据量为：{}".format(d["domain"],part,len(part_data)))
        statics_list = [(k,v) for k,v in statics.items()]
        statics_list.sort(key=lambda x:x[1],reverse=True)
        print("{} {} 标签统计信息：{}".format(d["domain"],part,statics_list))
    return part_data,d["domain"]
    
def domain_data():
    all_data = load_json(data_path)
    domain_d = defaultdict(list)
    for item in all_data:
        domain_d[item["domain"]].append(item)
    # split data for all domain
    all_data_train = []
    all_data_dev = []
    all_data_test = []
    all_attr = load_json("data/attrs.json")
    all_attr_set = set()
    for d,attr in all_attr.items():
        if d=="all":
            continue
        all_attr_set |= set(attr)
    attr2idx_all = {a:idx for idx,a in enumerate(all_attr_set)}
    for d,das in domain_d.items():
        attrs = load_json(f"data/{d}/attrs.json")
        attr2idx = {a:idx for idx,a in enumerate(attrs)}
        train_data, dev_data, test_data = split_data(das)
        _, domain = handle_data(train_data,"train",attr2idx)
        _, domain = handle_data(dev_data,"dev",attr2idx)
        _, domain = handle_data(test_data,"test",attr2idx)
        print("{}共有{}条数据".format(domain,len(train_data)+len(dev_data)+len(test_data)))
        train_data, domain = handle_data(train_data,"train",attr2idx_all,save=False)
        dev_data, domain = handle_data(dev_data,"dev",attr2idx_all,save=False)
        test_data, domain = handle_data(test_data,"test",attr2idx_all,save=False)
        all_data_train.extend(train_data)
        all_data_dev.extend(dev_data)
        all_data_test.extend(test_data)
    save_json(all_data_train, "data/all/{}.json".format("train"))
    save_json(all_data_dev, "data/all/{}.json".format("dev"))
    save_json(all_data_test, "data/all/{}.json".format("test"))
    print("all行业 train:{}, dev:{}, test:{}".format(len(all_data_train),len(all_data_dev),len(all_data_test)))

def new_domain_data():
    train_data_all = load_json(train_path)
    dev_data_all = load_json(dev_path)
    test_data_all = load_json(test_path)
    def get_domain_split(data):
        dict_data = defaultdict(list)
        for item in data:
            dict_data[item["domain"]].append(item)
        return dict_data
    dict_train_all = get_domain_split(train_data_all)
    dict_dev_all = get_domain_split(dev_data_all)
    dict_test_all = get_domain_split(test_data_all)
    domain_data_statics = defaultdict(dict)
    all_data_train = []
    all_data_dev = []
    all_data_test = []
    all_attr = load_json("data/attrs.json")
    all_attr_set = set()
    for d,attr in all_attr.items():
        if d=="all":
            continue
        all_attr_set |= set(attr)
    attr2idx_all = {a:idx for idx,a in enumerate(all_attr_set)}
    for d in dict_train_all:
        attrs = load_json(f"data/{d}/attrs.json")
        attr2idx = {a:idx for idx,a in enumerate(attrs)}
        train_data = dict_train_all[d]
        dev_data = dict_dev_all[d]
        test_data = dict_test_all[d]
        _, domain = handle_data(train_data,"train",attr2idx)
        _, domain = handle_data(dev_data,"dev",attr2idx)
        _, domain = handle_data(test_data,"test",attr2idx)
        print("{}共有{}条数据".format(domain,len(train_data)+len(dev_data)+len(test_data)))
        train_data, domain = handle_data(train_data,"train",attr2idx_all,save=False)
        dev_data, domain = handle_data(dev_data,"dev",attr2idx_all,save=False)
        test_data, domain = handle_data(test_data,"test",attr2idx_all,save=False)
        domain_data_statics[domain]["train"] = len(train_data)
        domain_data_statics[domain]["dev"] = len(dev_data)
        domain_data_statics[domain]["test"] = len(test_data)
        all_data_train.extend(train_data)
        all_data_dev.extend(dev_data)
        all_data_test.extend(test_data)
    print(domain_data_statics)
    save_json(all_data_train, "data/all/{}.json".format("train"))
    save_json(all_data_dev, "data/all/{}.json".format("dev"))
    save_json(all_data_test, "data/all/{}.json".format("test"))
    print("all行业 train:{}, dev:{}, test:{}".format(len(all_data_train),len(all_data_dev),len(all_data_test)))

def get_loss_weight(domain="all"):  
    class_weights = {}
    positive_weights = {}
    negative_weights = {}
    pos_weights = []
    data = load_json(f"data/{domain}/train.json")
    # data.extend(load_json(f"data/{domain}/dev.json"))
    # data.extend(load_json(f"data/{domain}/test.json"))
    all_attr = load_json("data/attrs.json")
    if domain=="all":
        label_dim = all_attr["all"]
    else:
        label_dim = len(all_attr[domain])
    sum_vec = np.zeros(label_dim)
    for d in data:
        sum_vec+=np.array(list(map(int,[a[0] for a in d["label"].split(",")])))
    
    N = len(data)
    for i in range(label_dim):
        if sum_vec[i]==0:
            positive_weights[f"label{i}"] = 0.5
            negative_weights[f"label{i}"] = 0.5
            pos_weights.append(5000)
        else:
            positive_weights[f"label{i}"] = N/(sum_vec[i]*2)
            negative_weights[f"label{i}"] = N/((N - sum_vec[i])*2)
            pos_weights.append(int((N - sum_vec[i])/sum_vec[i]))
    class_weights['positive_weights'] = positive_weights
    class_weights['negative_weights'] = negative_weights
    class_weights["pos_weights"] = pos_weights
    save_json(class_weights,f"data/{domain}/class_weights.json")
    
def get_statics(domain):
    statics = {}
    for domain in ch2en.keys():
        if domain=="all":
            continue
        label_statics = defaultdict(list)
        data = load_json(f"data/{domain}/train.json")
        for d in data:
            raw_label = list(d["raw_label"].keys())
            raw_label.sort()
            label_statics[" ".join(raw_label)].append(d["text"])
        label_statics_list = [(k,len(v),v) for k,v in label_statics.items()]
        label_statics_list.sort(key=lambda x:x[1], reverse=True)
        label_statics_plt = [[kk[0],kk[1]] for kk in label_statics_list][:10]
        save_json(label_statics_plt,f"data/statics/s_{domain}.json")
        statics[domain]=label_statics_plt
    print(statics)
    
    #可视化
    row = 1
    col = 1
    for domain,sta in statics.items():
        plt.subplot(int(f"8{row}{col}"))
        col+=2
        if col==5:
            row+=2
            col==1
        plt.bar([k[0] for k in sta],[k[1] for k in sta],color='b')
        plt.show()
    plt.savefig("data/statics/img.jpg")

def load_json(path):
    return json.load(open(path,'r',encoding='utf-8'))
def save_json(data,path):
    fpath = path[:path.rfind("/")]
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    json.dump(data,open(path,'w',encoding='utf-8'),indent=2,ensure_ascii=False)
    
"sos_c utter1 eos_c sos_a attr1 eos_a sos_c utter2 eos_c sos_a attr2 eos_a"
def trans():
    all_attr = json.load(open("data/all_attr.json",'r',encoding='utf-8'))
    lab_dim = len(all_attr)
    attr2idx = {k:idx for idx,k in enumerate(all_attr)}
    statics = {"attr":defaultdict(int),"text":defaultdict(int)}
    def trans_label(data,part):
        for d in data:
            statics["attr"][str(len(d["raw_label"].keys()))]+=1
            statics["text"][str(len(d["text"]))]+=1
            label_vec = np.zeros(lab_dim)
            for l in d["raw_label"].keys():
                if l in attr2idx:
                    label_vec[attr2idx[l]] = 1
            
            d["label"] = ",".join(map(str,label_vec))

        dump_json(f"data/{part}.json",data)
        
    train_data = load_json("data/train.json")
    dev_data = load_json("data/dev.json")
    test_data = load_json("data/test.json")
    trans_label(train_data,"train")
    trans_label(dev_data,"dev")
    trans_label(test_data,"test")
    for k in statics:
        statics[k] = [(k,v) for k,v in statics[k].items()]
        statics[k].sort(key=lambda x:x[1],reverse=True)
    print(statics["attr"][:10])


def get_domain_data():
    pass
    
if __name__ == "__main__":
    # seed = 42
    # get_attrs()
    # random.seed(seed)
    # domain_data()
    
    get__new_attrs()
    new_domain_data()
    
    # get_loss_weight()
    # get_statics("鞋类行业")
    