import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import os
import time
import logging

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INT = 0
LONG = 1
FLOAT = 2
EOS = 2
def save_json(data,path):
    json.dump(data, open(path,'r',encoding="utf-8"),indent=2,ensure_ascii=False)

def load_json(path):
    json.load(open(path,'r',encoding='utf-8'))

def focal_loss(logits, labels, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.

    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).

    Args:
      logits: A float tensor of size [batch, num_classes].
      labels: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.

    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    bce_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits)))

    loss = modulator * bce_loss

    # weighted_loss = alpha * loss
    weighted_loss = loss
    loss = torch.sum(weighted_loss)
    loss /= torch.sum(labels)
    return loss


def penalty_ce_loss(pred, y, penalty, epoch):
  # l = -[a*y*log(x)+(1-y)*log(1-x)]
  if penalty/epoch>1:
    penalty = penalty/epoch
  else:
    penalty = 1
  return -(penalty * pred.log()*y + (1-y)*(1-pred).log()).mean()
  

def attr_seq(attr, attr2idx, max_len=7):
  a_seq = []
  for a in attr:
      a_seq.append(attr2idx[a]+3)
  # if len(act_seq)==0:
  #     da = 'general-unk'
  #     act_seq.append(config.da2idx[da])
  a_seq = sorted(a_seq)
  a_seq.insert(0, 1) # SOS
  a_seq.append(2)  # EOS
  a_seq = pad_to(max_len, a_seq, True)   
  return a_seq

def pad_to(max_len, tokens, do_pad=True):
    if len(tokens) >= max_len:
        return tokens[0:max_len - 1] + [tokens[-1]]
    elif do_pad:
        return tokens + [0] * (max_len - len(tokens)) #PAD
    else:
        return tokens

class NLLEntropy(_Loss):

    def __init__(self, padding_idx, avg_type=None):
        super(NLLEntropy, self).__init__()
        self.padding_idx = padding_idx
        self.weight = None
        self.avg_type = avg_type

    def forward(self, net_output, labels=None):
        batch_size = net_output.size(0)
        input = net_output.view(-1, net_output.size(-1))
        target = labels.view(-1)

        if self.avg_type is None:
            loss = F.nll_loss(input, target, size_average=False,
                              ignore_index=self.padding_idx,
                              weight=self.weight)
        elif self.avg_type == 'seq':
            loss = F.nll_loss(input, target, size_average=False,
                              ignore_index=self.padding_idx,
                              weight=self.weight)
            loss = loss / batch_size
        elif self.avg_type == 'real_word':
            loss = F.nll_loss(input, target, size_average=True,
                              ignore_index=self.padding_idx,
                              weight=self.weight, reduce=False)
            loss = loss.view(-1, net_output.size(1))
            loss = torch.sum(loss, dim=1)
            word_cnt = torch.sum(torch.sign(labels), dim=1).float()
            loss = loss/word_cnt
            loss = torch.mean(loss)
        elif self.avg_type == 'word':
            loss = F.nll_loss(input, target, size_average=True,
                              ignore_index=self.padding_idx,
                              weight=self.weight)
        return loss
    
def prepare_input(data):
    """
    Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
    """
    if isinstance(data, Mapping):
        return type(data)({k: prepare_input(v) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(prepare_input(v) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(DEVICE)
    return data

def cast_type(var, dtype, use_gpu):
    if use_gpu:
        if dtype == INT:
            var = var.type(torch.cuda.IntTensor)
        elif dtype == LONG:
            var = var.type(torch.cuda.LongTensor)
        elif dtype == FLOAT:
            var = var.type(torch.cuda.FloatTensor)
        else:
            raise ValueError("Unknown dtype")
    else:
        if dtype == INT:
            var = var.type(torch.IntTensor)
        elif dtype == LONG:
            var = var.type(torch.LongTensor)
        elif dtype == FLOAT:
            var = var.type(torch.FloatTensor)
        else:
            raise ValueError("Unknown dtype")
    return var

def init_logging_handler(log_dir, extra=''):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    stderr_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('{}/log_{}.txt'.format(log_dir, current_time+extra))
    logging.basicConfig(handlers=[stderr_handler, file_handler])
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)