import logging
from omegaconf import OmegaConf
import pandas as pd
from typing import Union, List, Iterable, Dict
import numpy as np
import torch
from torch import embedding, nn, Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset
from data_process import load_json
import random
from utils import definitions
import sys

from tqdm import tqdm

from transformers import BertPreTrainedModel, BertModel, PreTrainedTokenizer, BertTokenizer
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, IntervalStrategy
from transformers import AdamW
from torch.optim import Adam

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import precision_score,recall_score,f1_score
from utils.utils import focal_loss, penalty_ce_loss, init_logging_handler
from data_process import ch2en


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiLabelDataset(Dataset):
    def __init__(self,
                 examples,
                 tokenizer: PreTrainedTokenizer,
                 max_length: int = 64):
        
        self.tokenizer = tokenizer
        self.examples = []
        for example in tqdm(examples,
                            desc="tokenize for multilabel dataset"):
            self.examples.append(
                self.tokenizer(
                    example["text"],
                    padding="max_length",
                    truncation=True,
                    max_length=max_length),
                )
            self.examples[-1]["label_ids"] = torch.tensor(example["label"], dtype=torch.long)
        
    def __getitem__(self, item):
        """
        :return List[List[str]]
        """
        features = self.examples[item]
        return features

    def __len__(self):
        return len(self.examples)

class BertForMultiLabelNet(BertPreTrainedModel):
    # _keys_to_ignore_on_load_unexpected = [
    #     "output_layer.weight",
    #     "output_layer.bias"
    # ]
    def __init__(self,config, cfg):
        super().__init__(config)
        self.cfg = cfg
        self.bert = BertModel(config)
        self.output_layer = nn.Sequential(nn.Linear(cfg["b_dim"], cfg["h_dim"]),
                                 nn.ReLU(),
                                 nn.Linear(cfg["h_dim"], cfg["h_dim"]),
                                 nn.ReLU(),
                                 nn.Linear(cfg["h_dim"], cfg["label_num"]))
        self.sigmoid = nn.Sigmoid()
        self.multi_entropy_loss = nn.BCELoss()
        self.weight = torch.tensor([0.5,1])
        self.fl2 = focal_loss
        self.pnloss = penalty_ce_loss
        # self.class_weights = load_json(cfg["class_weights"])
        # self.class_weights = torch.tensor(self.class_weights["pos_weights"], dtype=torch.long).cuda()
        
        
    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                labels=None):
        output = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            output_hidden_states=self.cfg["output_hidden_states"])
        # emb = self.layer1(output[1])
        emb = self.output_layer(output[1])
        logits = self.sigmoid(emb)

        outputs = (
            logits,
            output[2],
        ) if self.cfg["output_hidden_states"] else (logits, )

        if labels is not None:
            labels = labels.to(torch.float)
            # loss = self.multi_entropy_loss(logits, labels)
            loss = self.pnloss(logits, labels, self.cfg["penalty"], 1)
            outputs = (loss, ) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):    
        labels = inputs.get("labels").to(torch.float)
        # forward pass
        outputs = model(**inputs)
        logits = outputs[1]
        # compute custom loss (suppose one has 3 labels with different weights)
        # loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0]))
        # loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        penalty = 3
        # epoch = self.state.epoch
        # if epoch==0:
        #     penalty = penalty
        # elif penalty/epoch>1:
        #     penalty = penalty/epoch
        # else:
        #     penalty = 1
        loss = -(penalty * logits.log()*labels + (1-labels)*(1-logits).log()).mean()
        # loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='mean', pos_weight=model.class_weights)
        return (loss, outputs) if return_outputs else loss
    
def compute_metrics(pred):
    labels = pred.label_ids
    scores = np.where(pred.predictions>0.5,1,0)
    precision = precision_score(labels,scores,average='samples')
    recall = recall_score(labels,scores,average='samples')
    f1 = f1_score(labels,scores,average='samples')
    # n,m = scores.shape
    # TP = np.sum(np.multiply(labels,scores))     #predict和label都为1的数目
    # assert np.sum(scores)!=0, "logits is {}".format(np.sum(scores))
    # precision = 0
    # if np.sum(scores)!=0:
    #     precision = TP/np.sum(scores)           #预测正确除以全部预测为真的数目
    # recall = TP/np.sum(labels)                  #预测正确除以label中为1的数目
    # f1 = 2 * precision * recall / (precision + recall)
            
    return {'f1': f1, 'precision': precision, 'recall': recall}

def load_data_from_json(path):
    data = load_json(path)
    for d in data:
        d["label"] = list(map(int,[a[0] for a in d["label"].split(",")]))
    return data

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dm', type=str, default="all", help='identify the domain')
    parser.add_argument('-test', type=bool, default=False, help='identify the domain')
    parser.add_argument('-ckpt', type=str, default="", help='the path of model')
    parser.add_argument('-suffix', type=str, default="", help='identify the function')
    return parser

def main(conf):
    parser = get_parser()
    argv = sys.argv[1:]
    args, _ = parser.parse_known_args(argv)
    all_domain = load_json("data/all_domain.json")
    all_attrs = load_json("data/attrs.json")
    # args.dm = "all"
    if args.dm in all_domain+["all"]:
        conf["train_path"] = conf["train_path"].replace("train",f"/{args.dm}/train")
        conf["dev_path"] = conf["dev_path"].replace("dev",f"/{args.dm}/dev")
        conf["test_path"] = conf["test_path"].replace("test",f"/{args.dm}/test")
        conf["class_weights"] = f"data/{args.dm}/class_weights.json"
        if args.suffix:
            conf["output_dir"]+="/"+args.suffix
            conf["log_dir"]+="/"+args.suffix
            conf["logging_dir"]+="/"+args.suffix
        conf["output_dir"]+="/"+args.dm
        conf["log_dir"]+="/"+args.dm
        conf["logging_dir"]+="/"+ch2en[args.dm]
        
        print(f"当前数据为：{args.dm}")
    else:
        raise NotADirectoryError()
    if args.dm=="all":
        conf["label_num"] = all_attrs["all"]
    else:
        conf["label_num"] = len(all_attrs[args.dm])
    model_path = conf["model_name_or_path"] if conf["do_train"] and not args.test else conf["output_dir"] + "/"+ args.ckpt
    init_logging_handler(conf["logging_dir"])
    logging.info(conf)
    logging.info("load model from: "+model_path)
    if not args.test and args.dm !="all":
        all_model_path = conf["output_dir"] + "/"+ args.ckpt
        all_model_path = all_model_path.replace(args.dm,"all")
        logging.info(f"load model from pretrain all domain: {all_model_path}")
        model = BertForMultiLabelNet.from_pretrained(all_model_path,conf,ignore_mismatched_sizes=True)
    else:
        model = BertForMultiLabelNet.from_pretrained(model_path,conf,ignore_mismatched_sizes=True)
    tokenizer = BertTokenizer.from_pretrained(conf["model_name_or_path"])
    all_special_token = definitions.SPE_TOKEN
    tokenizer.add_special_tokens({'additional_special_tokens':all_special_token})
    model.resize_token_embeddings(len(tokenizer))
    no_decay = ['bias', 'gamma', 'beta']
    parameters = [
            {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0},
            {'params': [p for n, p in model.named_parameters() if "bert" not in n],
             'lr': 1e-3,
             'weight_decay_rate':0.01}
        ]
    kwargs = {
        'betas': (0.9, 0.999),
        'eps': 1e-08
    }
    optimizer = AdamW(parameters, lr=5e-5, **kwargs)
    # scheduler = get_constant_schedule(optimizer)
    
    logging.info('*** TRAIN ***')
    # 读取训练集和开发集
    logging.info('Loading train/dev dataset')
    train_dataset = MultiLabelDataset(load_data_from_json(conf["train_path"]),
                                    tokenizer, conf["max_length"])
    dev_dataset = MultiLabelDataset(load_data_from_json(conf["dev_path"]),
                                    tokenizer, conf["max_length"])
    test_dataset = MultiLabelDataset(load_data_from_json(conf["test_path"]),
                                    tokenizer, conf["max_length"])
    save_steps = len(train_dataset)//conf["batch_size"]
    logging.info(f"save_step:{save_steps}")
    logging.info("train:{} dev:{} test:{}".format(len(train_dataset),len(dev_dataset),len(test_dataset)))
    training_args = TrainingArguments(
        output_dir=conf["output_dir"],          # output directory
        num_train_epochs=conf["num_train_epochs"],              # total number of training epochs
        per_device_train_batch_size=conf["batch_size"],  # batch size per device during training
        per_device_eval_batch_size=conf["batch_size"],   # batch size for evaluation
        warmup_ratio=0.2,                # ratio of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir=conf["log_dir"],            # directory for storing logs
        logging_steps=50,
        # evaluation_strategy="epoch",
        evaluation_strategy="steps",
        # save_strategy ="epoch",
        save_strategy ="steps",
        save_steps=save_steps,
        eval_steps=save_steps,
        do_train=True,
        do_eval=True,
        load_best_model_at_end =True,
        metric_for_best_model ="eval_f1",
        greater_is_better=True,
        save_total_limit = 10
    )
    trainer = CustomTrainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=dev_dataset,             # evaluation dataset
        compute_metrics=compute_metrics,
        optimizers=[optimizer,None],
        callbacks = [EarlyStoppingCallback(early_stopping_patience=10)]
    )
    
    if not args.test:
        train_metrics = trainer.train()
        trainer.save_model()
        logging.info(trainer.state.log_history)
    test_metrics = trainer.predict(test_dataset)
    logging.info(test_metrics)

if __name__ == '__main__':
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    conf = OmegaConf.load('config.yaml')
    main(conf)
