from torch.utils.data import DataLoader, Dataset
import logging
from omegaconf import OmegaConf
import pandas as pd
from typing import Union, List, Iterable, Dict
import numpy as np
import torch
from torch import embedding, nn, Tensor
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset
from data_process import load_json
import random
from utils import definitions
from utils.utils import NLLEntropy, prepare_input, cast_type, init_logging_handler
import sys
import time
from tqdm import tqdm
import os
from transformers import BertPreTrainedModel, BertModel, PreTrainedTokenizer, BertTokenizer
from transformers import Trainer, TrainingArguments
from transformers import AdamW
from transformers.data import data_collator
from torch.autograd import Variable
from sklearn.metrics import precision_score,recall_score,f1_score
from data_process import ch2en, en2ch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_GPU = True if torch.cuda.is_available() else False


class DatasetSeq(Dataset):
    def __init__(self, s_s, a_seq, tokenizer,max_length=128):
        self.tokenizer = tokenizer
        self.s_s = []
        for idx,s in tqdm(enumerate(s_s),desc="tokenize for multilabel dataset"):
            self.s_s.append(
                self.tokenizer(
                    s,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length),
                )
            self.s_s[-1]["label_ids"]=torch.tensor(a_seq[idx], dtype=torch.long)
        self.num_total = len(s_s)
        # print(self.s_s[:5])
        # a = np.zeros(20)
        # a_list = defaultdict(int)
        # for a_l in a_s:
        #     a[int(a_l.sum().item())]+=1
        #     key = str(a_l.tolist())
        #     a_list[key] += 1
        # k=0
        # for w in sorted(a_list, key=a_list.get, reverse=True):
        #     logging.info("{}: {}".format(w, a_list[w]))
        #     logging.info(k)
        #     k+=1
        # logging.info(a)
        # logging.info(len(a_list))
    
    def __getitem__(self, index):
        s = self.s_s[index]
        # a_seq = self.a_seq[index]
        return s
    
    def __len__(self):
        return self.num_total   

class DiaSeq(object):
    def __init__(self, cfg):
        self.cfg= cfg
        # self.policy_clip = args.policy_clip
    
        self.policy = MultiDiscretePolicy(cfg).to(device=DEVICE)
        # logging.info(summary(self.policy, show_weights=False))
        # logging.info(summary(self.value , show_weights=False))
        self.tokenizer = BertTokenizer.from_pretrained(conf["model_name_or_path"])
        def get_data(path):
            data = load_json(path)
            s = [d["text"] for d in data]
            a = [d["attr_seq"] for d in data]
            dataset = DatasetSeq(s,a, self.tokenizer)
            dataloader = DataLoader(dataset, self.cfg.batch_size, True,collate_fn = data_collator.torch_default_data_collator)
            return dataloader,len(dataset)
        
        self.data_train, train_num = get_data(cfg["train_path"])
        self.data_valid, valid_num = get_data(cfg["dev_path"])
        self.data_test, test_num = get_data(cfg["test_path"])
        logging.info("train:{} valid:{} test:{}".format(train_num,valid_num,test_num))
        self.print_per_batch = train_num//cfg.batch_size  #args.print_per_batch
        self.save_dir = cfg.gen_output_dir
        self.save_per_epoch = 1 # args.save_per_epoch
        self.policy.eval()
        # no_decay = ['bias', 'gamma', 'beta']
        # parameters = [
        #     {'params': [p for n, p in self.policy.bert.named_parameters() if not any(nd in n for nd in no_decay)],
        #         'weight_decay_rate': 0.01},
        #     {'params': [p for n, p in self.policy.bert.named_parameters() if any(nd in n for nd in no_decay)],
        #         'weight_decay_rate': 0.0},
        #     {'params': [p for n, p in self.policy.named_parameters() if "bert" not in n],
        #         'lr': 1e-3,
        #         'weight_decay_rate':0.01}
        # ]
        # kwargs = {
        #     'betas': (0.9, 0.999),
        #     'eps': 1e-08
        # }
        # self.policy_optim = AdamW(parameters, lr=conf.backbone_lr, **kwargs)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=conf.backbone_lr)
        self.epoch = 0
        self.best_valid_loss = np.inf
        self.valid_loss_threshold = np.inf
        self.patience = 10

    def update_loop(self, data):
        # gen_type: beam, greedy
        # mode: GEN, TEACH_FORCE
        # s, a, d, a_seq = self.retrieve_expert(True)
        # print(s)
        # print(a_seq)
        loss = self.policy(**data, mode=TEACH_FORCE)
        return loss
    
    def train(self, epoch):
        self.policy.train()
        a_loss = 0.
        for i, data in enumerate(self.data_train):
            self.policy_optim.zero_grad()
            loss_a = self.update_loop(data)
            a_loss += loss_a.item()
            loss_a.backward()
            # torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.policy_clip)
            self.policy_optim.step()
            
            if (i+1) % self.print_per_batch == 0:
                a_loss /= self.print_per_batch
                logging.info(' epoch {}, iter {}, loss_a:{}'.format(epoch, i, a_loss))
                a_loss = 0.

        self.policy.eval()
        if (epoch+1) % self.save_per_epoch == 0:
            self.save(self.save_dir, epoch, True)

        for i, data in enumerate(self.data_valid):
            loss_a = self.update_loop(data)
            a_loss += loss_a.item()
        valid_loss = a_loss/len(self.data_valid)
        logging.info(' validation, epoch {}, loss_a:{}'.format(epoch, valid_loss))
        if valid_loss < self.best_valid_loss:
            if valid_loss <= self.valid_loss_threshold * self.cfg.improve_threshold:
                self.patience = max(self.patience,
                                epoch * self.cfg.patient_increase)
                self.valid_loss_threshold = valid_loss
                logging.info("Update patience to {}".format(self.patience))
            self.best_valid_loss = valid_loss

            logging.info(' best model saved')
            self.save(self.save_dir, 'best', True)
        

        if self.cfg.early_stop and self.patience <= epoch:
            if epoch < self.cfg.num_train_epochs:
                logging.info("!!Early stop due to run out of patience!!")

            logging.info("Best validation loss %f" % self.best_valid_loss)
            return True
        return False

    def evaluate(self,type="valid"):
        self.policy.eval()
        preds = []
        labels = []
        data_evaluate = self.data_valid
        if type=="test":
            data_evaluate = self.data_test
        for i, data in enumerate(data_evaluate):
            pred,label = self.policy.select_action(**data)
            # print(pred)
            # print(label)
            preds.extend(pred)
            labels.extend(label)
        precision = precision_score(labels,preds,average='samples')
        recall = recall_score(labels,preds,average='samples')
        f1 = f1_score(labels,preds,average='samples')
        logging.info('f1:{} precision:{} recal:{}'.format(f1,precision,recall))
            
    def save(self, directory, epoch, rl_only=False):
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        # if not rl_only:
            # self.rewarder.save_irl(directory, epoch)

        torch.save(self.policy.state_dict(), directory + '/' + str(epoch) + '.bin')

        logging.info(' epoch {}: saved network to mdl'.format(epoch))

    def load(self, filename, only_bert=False):
        policy_mdl = filename + '.bin'
        if os.path.exists(policy_mdl):
            load_dict = {}
            if only_bert:
                model_dict = torch.load(policy_mdl)
                load_dict= {k:v for k,v in model_dict.items() if "bert" in k}
            else:
                load_dict = torch.load(policy_mdl)
            self.policy.load_state_dict(load_dict,False)
            logging.info(' loaded checkpoint from file: {}'.format(policy_mdl))
        
        best_pkl = filename + '.pkl'
        if os.path.exists(best_pkl):
            with open(best_pkl, 'rb') as f:
                best = pickle.load(f)
        else:
            best = [float('inf'),float('inf'),float('-inf')]
        return best

INT = 0
LONG = 1
FLOAT = 2
EOS = 2
TEACH_FORCE = "teacher_forcing"
TEACH_GEN = "teacher_gen"
GEN = "gen"
GEN_VALID = 'gen_valid'

class BaseRNN(nn.Module):
    # SYM_MASK = PAD
    SYM_EOS = EOS

    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'
    KEY_LATENT = 'latent'
    KEY_RECOG_LATENT = 'recog_latent'
    KEY_POLICY = "policy"
    KEY_G = 'g'
    KEY_PTR_SOFTMAX = 'ptr_softmax'
    KEY_PTR_CTX = "ptr_context"


    def __init__(self, vocab_size, input_size, hidden_size, input_dropout_p,
                 dropout_p, n_layers, rnn_cell, bidirectional):
        super(BaseRNN, self).__init__()
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_dropout_p = input_dropout_p
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        self.dropout_p = dropout_p
        self.rnn = self.rnn_cell(input_size, hidden_size, n_layers,
                                 batch_first=True, dropout=dropout_p,
                                 bidirectional=bidirectional)
        if rnn_cell.lower() == 'lstm':
            for names in self.rnn._all_weights:
                for name in filter(lambda n: "bias" in n, names):
                    bias = getattr(self.rnn, name)
                    n = bias.size(0)
                    start, end = n // 4, n // 2
                    bias.data[start:end].fill_(1.)

    def gumbel_max(self, log_probs):
        """
        Obtain a sample from the Gumbel max. Not this is not differentibale.
        :param log_probs: [batch_size x vocab_size]
        :return: [batch_size x 1] selected token IDs
        """
        sample = torch.Tensor(log_probs.size()).uniform_(0, 1)
        sample = cast_type(Variable(sample), FLOAT, self.use_gpu)

        # compute the gumbel sample
        matrix_u = -1.0 * torch.log(-1.0 * torch.log(sample))
        gumbel_log_probs = log_probs + matrix_u
        max_val, max_ids = torch.max(gumbel_log_probs, dim=-1, keepdim=True)
        return max_ids

    def repeat_state(self, state, batch_size, times):
        new_s = state.repeat(1, 1, times)
        return new_s.view(-1, batch_size * times, self.hidden_size)

    def forward(self, *args, **kwargs):
        raise NotImplementedError()
    
class DecoderRNN(BaseRNN):
    def __init__(self, vocab_size, max_len, input_size, hidden_size, sos_id,
                 eos_id, n_layers=1, rnn_cell='lstm', input_dropout_p=0,
                 dropout_p=0, use_attention=False, use_gpu=True, embedding=None, output_size=None,
                 tie_output_embed=False, cat_mlp=False):

        super(DecoderRNN, self).__init__(vocab_size, input_size,
                                         hidden_size, input_dropout_p,
                                         dropout_p, n_layers, rnn_cell, False)

        self.output_size = vocab_size if output_size is None else output_size
        self.max_length = max_len
        self.use_attention = use_attention
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.init_input = None
        self.use_gpu = use_gpu
        self.cat_mlp = cat_mlp

        if embedding is None:
            self.embedding = nn.Embedding(vocab_size, self.input_size)   #(170,64)
        else:
            self.embedding = embedding
        self.project = nn.Linear(self.hidden_size, self.output_size)    #(100,170)
        self.function = F.log_softmax

    def forward_step(self, input_var, hidden, encoder_outputs, h_0):
        # (10,1)
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        
        # [batch_size, output_size(dec_init_state+action_dim)]
        embedded = self.embedding(input_var)    #编码动作
        if self.cat_mlp:                        #拼接对话状态
            embedded = torch.cat([embedded, h_0], -1)       #(10,1,164)

        embedded = self.input_dropout(embedded)

        output, hidden = self.rnn(embedded, hidden)     #(10,1,100)  (1,10,100)

        attn = None

        output = output.contiguous()
        logits = self.project(output.view(-1, self.hidden_size))    #每个时间步进行解码     # (10,100)-->(10,170)
        predicted_softmax = self.function(logits, dim=logits.dim()-1).view(batch_size, output_size, -1)     #(10,1,170)
        return predicted_softmax, hidden, attn

    def forward(self, batch_size, inputs=None, init_state=None,
                attn_context=None, mode=TEACH_FORCE, gen_type='greedy',
                beam_size=4):

        # sanity checks
        ret_dict = dict()
        h_0 = init_state.squeeze(0).unsqueeze(1).repeat(1,self.max_length-1, 1)     #(1,batch_size,100)-->(batch_size,1,100)-->(batch_size,max_length-1,100) (2,19,100)
        if mode == GEN:
            inputs = None

        if gen_type != 'beam':
            beam_size = 1

        if inputs is not None:
            decoder_input = inputs
        else:
            # prepare the BOS inputs
            with torch.no_grad():
                bos_var = Variable(torch.LongTensor([self.sos_id])) 
            bos_var = cast_type(bos_var, LONG, self.use_gpu)
            decoder_input = bos_var.expand(batch_size*beam_size, 1)     #起始动作   (10,1)
            # logging.info("dec input: {}".format(decoder_input.shape))
            h_0 = init_state.squeeze(0).unsqueeze(1).repeat(beam_size, 1, 1)  # 12, 1, 100  #(10,1,100)
            # logging.info("h0: {}".format(h_0.shape))

        if mode == GEN and gen_type == 'beam':
            # if beam search, repeat the initial states of the RNN
            if self.rnn_cell is nn.LSTM:
                h, c = init_state
                decoder_hidden = (self.repeat_state(h, batch_size, beam_size),
                                  self.repeat_state(c, batch_size, beam_size))
            else:
                decoder_hidden = self.repeat_state(init_state,
                                                   batch_size, beam_size)       #(1,10,100)  10=2*5
        else:
            decoder_hidden = init_state

        decoder_outputs = [] # a list of logprob
        sequence_symbols = [] # a list word ids
        back_pointers = [] # a list of parent beam ID
        lengths = np.array([self.max_length] * batch_size * beam_size)      #[20,...] dim=(10,)

        def decode(step, cum_sum, step_output, step_attn):
            decoder_outputs.append(step_output)         #(10,1,170)
            step_output_slice = step_output.squeeze(1)  #(10,170)

            if self.use_attention:
                ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)

            if gen_type == 'greedy':
                symbols = step_output_slice.topk(1)[1]
            elif gen_type == 'sample':
                symbols = self.gumbel_max(step_output_slice)
            elif gen_type == 'beam':
                if step == 0:
                    seq_score = step_output_slice.view(batch_size, -1)  #(2,850)
                    seq_score = seq_score[:, 0:self.output_size]        #(2,170)        # 第一次解码只需保留前(batch_size,170)，因为初始动作为SOS，解码出来的取概率最大的动作即可
                else:
                    seq_score = cum_sum + step_output_slice
                    seq_score = seq_score.view(batch_size, -1)

                top_v, top_id = seq_score.topk(beam_size)

                back_ptr = top_id.div(self.output_size).view(-1, 1)     # 对id除以170，beam为5，表示取第几个（前一时间步延申出哪个概率值最大）
                symbols = top_id.fmod(self.output_size).view(-1, 1)     # 当前第几个action，[0,...,169,170,...,339]
                cum_sum = top_v.view(-1, 1)
                back_pointers.append(back_ptr)
            else:
                raise ValueError("Unsupported decoding mode")

            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > di) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return cum_sum, symbols

        # Manual unrolling is used to support random teacher forcing.
        # If teacher_forcing_ratio is True or False instead of a probability,
        # the unrolling can be done in graph
        if mode == TEACH_FORCE:
            decoder_output, decoder_hidden, attn = self.forward_step(
                decoder_input, decoder_hidden, attn_context, h_0)

            # in teach forcing mode, we don't need symbols.
            decoder_outputs = decoder_output

        else:
            # do free running here
            cum_sum = None
            for di in range(self.max_length):
                decoder_output, decoder_hidden, step_attn = self.forward_step(      #decoder_output为每个时间步的logits
                    decoder_input, decoder_hidden, attn_context, h_0)

                cum_sum, symbols = decode(di, cum_sum, decoder_output, step_attn)
                decoder_input = symbols

            decoder_outputs = torch.cat(decoder_outputs, dim=1)

            if gen_type == 'beam':
                # do back tracking here to recover the 1-best according to
                # beam search.
                final_seq_symbols = []
                cum_sum = cum_sum.view(-1, beam_size)   #beam_search最终的分数
                max_seq_id = cum_sum.topk(1)[1].data.cpu().view(-1).numpy()     #选择分数最大的下标（叶子节点）往前复原整个序列
                rev_seq_symbols = sequence_symbols[::-1]
                rev_back_ptrs = back_pointers[::-1]
                # print("rev_seq_symbols",rev_seq_symbols)
                # print("rev_back_ptrs",rev_back_ptrs)
                for symbols, back_ptrs in zip(rev_seq_symbols, rev_back_ptrs):
                    symbol2ds = symbols.view(-1, beam_size)     #(batch_size,beam_size)
                    back2ds = back_ptrs.view(-1, beam_size)     #(batch_size,beam_size)
                    # logging.info(symbol2ds)
                    selected_symbols = []
                    selected_parents =[]
                    for b_id in range(batch_size):
                        # selected_parents.append(back2ds[b_id, max_seq_id[b_id]])
                        selected_parents.append(back2ds[b_id, int(max_seq_id[b_id])])
                        selected_symbols.append(symbol2ds[b_id, int(max_seq_id[b_id])])
                    # logging.info(selected_symbols)
                    final_seq_symbols.append(torch.stack(selected_symbols, 0).unsqueeze(1))
                    max_seq_id = torch.stack(selected_parents).data.cpu().numpy()
                sequence_symbols = final_seq_symbols[::-1]

        # save the decoded sequence symbols and sequence length
        ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
        ret_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, decoder_hidden, ret_dict

class MultiDiscretePolicy(nn.Module):
    def __init__(self, cfg):
        super(MultiDiscretePolicy, self).__init__()
        self.cfg = cfg
        self.test_gentype = cfg.test_gentype
        self.decoder_hidden = cfg.h_dim//2
        self.bert = BertModel.from_pretrained(self.cfg["model_name_or_path"])
        self.net = nn.Sequential(nn.Linear(cfg.b_dim, cfg.h_dim),
                                nn.ReLU(),
                                nn.Linear(cfg.h_dim, self.decoder_hidden),  #(561,200,100)
                                )
        if self.cfg.cat_mlp:    # 每个时间步输入是否拼接初始编码
            decoder_input_size = cfg.embed_size + self.decoder_hidden   #(64+100)
        else:
            decoder_input_size = cfg.embed_size
        self.go_id = 1
        self.eos_id = 2
        self.embedding = nn.Embedding(cfg["label_num"], cfg.embed_size,
                                      padding_idx=0)
        self.decoder = DecoderRNN(cfg["label_num"], cfg["a_maxlen"],
                            decoder_input_size, self.decoder_hidden,
                            self.go_id, self.eos_id,
                            n_layers=1, rnn_cell='gru',
                            input_dropout_p=0.,
                            dropout_p=0.,
                            use_gpu=USE_GPU,
                            embedding=self.embedding,
                            cat_mlp=self.cfg.cat_mlp)
                            
        self.nll_loss = NLLEntropy(0, avg_type=self.cfg.avg_type)

    def forward(self, input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                labels=None, mode=None, gen_type=None):
        a_seq = labels
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        token_type_ids = token_type_ids.to(DEVICE)
        a_seq = a_seq.to(DEVICE)
        batch_size = len(input_ids)
        # logging.info("s shape: {}".format(s.shape))
        bert_output = self.bert(input_ids,attention_mask,token_type_ids)
        dec_init_state = self.net(bert_output[1]).unsqueeze(0)   #(1,batch_size,100)
        # logging.info("h: {}, inp: {}".format(dec_init_state.shape, a_seq.shape))
        dec_outs, dec_last, dec_ctx = self.decoder(batch_size, a_seq[:, 0:-1], dec_init_state,
                                            mode=mode, gen_type=gen_type,
                                            beam_size=1)
        labels = a_seq[:, 1:].contiguous()
        enc_dec_nll = self.nll_loss(dec_outs, labels)
        return enc_dec_nll

    def select_action(self, input_ids=None,
                attention_mask=None,
                token_type_ids=None, 
                labels=None,sample=True):
        # here we repeat the state twice to avoid potential errors in Decoder
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        token_type_ids = token_type_ids.to(DEVICE)
        a_seq = None
        batch_size = len(input_ids)
        # logging.info("s shape: {}".format(s.shape))
        bert_output = self.bert(input_ids,attention_mask,token_type_ids)
        dec_init_state = self.net(bert_output[1]).unsqueeze(0)   #(1,batch_size,100)
        dec_outs, dec_last, dec_ctx = self.decoder(batch_size, a_seq, dec_init_state,
                                            mode=GEN, gen_type=self.test_gentype,
                                            beam_size=5)
        pred_labels = [t.cpu().data.numpy() for t in dec_ctx[DecoderRNN.KEY_SEQUENCE]]
        pred_labels = np.array(pred_labels, dtype=int).squeeze(-1).swapaxes(0,1)
        # pred_labels = pred_labels[0]
        acts = np.zeros((batch_size,self.cfg.label_num-3))
        for bid in range(len(pred_labels)):
            for x in pred_labels[bid]:
                if x not in [0, 1, 2, 169]:
                    acts[bid][x-3]=1.
                elif x == 2:
                    break
        # logging.info(act)
        # 将label也组织成one-hot向量，方便计算f1
        labels_one_hot = np.zeros((batch_size,self.cfg.label_num-3))
        for bid in range(len(labels)):
            for x in labels[bid]:
                if x not in [0, 1, 2, 169]:
                    labels_one_hot[bid][x-3]=1.
                elif x == 2:
                    break
        return acts,labels_one_hot

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
    args.dm = en2ch[args.dm]
    if args.dm in all_domain+["all"]:
        conf["train_path"] = conf["train_path"].replace("train",f"/{args.dm}/train")
        conf["dev_path"] = conf["dev_path"].replace("dev",f"/{args.dm}/dev")
        conf["test_path"] = conf["test_path"].replace("test",f"/{args.dm}/test")
        conf["class_weights"] = f"data/{args.dm}/class_weights.json"
        if args.suffix:
            conf["gen_output_dir"]+="/"+args.suffix
            conf["gen_logging_dir"]+="/"+args.suffix
        conf["gen_output_dir"]+="/"+args.dm
        conf["gen_logging_dir"]+="/"+ch2en[args.dm]
        
        print(f"当前数据为：{args.dm}")
    else:
        raise NotADirectoryError()
    if args.dm=="all":
        conf["label_num"] = all_attrs["all"]+3
    else:
        conf["label_num"] = len(all_attrs[args.dm])+3
    
    # conf["model_name_or_path"] = conf["model_name_or_path"] if not args.test else conf["gen_output_dir"] + "/"+ args.ckpt
    init_logging_handler(conf["gen_logging_dir"])
    logging.info(conf)
    if not args.test:
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        # dir_name = datetime.now().isoformat()
        args.save_dir = os.path.join(conf["gen_output_dir"], args.suffix)
        logging.info(args.save_dir)
        
        # current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        logging.info('train {}'.format(current_time))
    
        agent = DiaSeq(conf)
        if args.dm != "all":
            logging.info("load checkpoint from pretrain all domain")
            agent.load(os.path.join(conf["gen_output_dir"].replace(args.dm,"all"),"best"),only_bert=True)
            agent.evaluate()
        for i in range(conf.num_train_epochs):
            if i%4==1:
                agent.evaluate()
            f = agent.train(i)
            current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            logging.info('epoch {} {}'.format(i, current_time))
            if f:
                break

        logging.info("############## Start Evaluating ##############")
        agent_test = DiaSeq(conf)
        agent_test.load(conf["gen_output_dir"]+'/best')
        logging.info("model loading finish and start evaluating")
        agent_test.evaluate()
        
    logging.info("############## Start Test ##############")
    agent_test = DiaSeq(conf)
    agent_test.load(os.path.join(conf["gen_output_dir"], "best"))
    logging.info("model loading finish and start evaluating")
    agent_test.evaluate("test")
    
if __name__ == "__main__":
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    conf = OmegaConf.load('config.yaml')
    print(conf)
    main(conf)