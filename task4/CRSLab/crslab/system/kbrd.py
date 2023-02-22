# -*- encoding: utf-8 -*-
# @Time    :   2020/12/4
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

# UPDATE
# @Time    :   2021/1/3
# @Author  :   Xiaolei Wang
# @email   :   wxl1999@foxmail.com

import os
import time
import torch
import wandb
from loguru import logger

from crslab.evaluator.metrics.base import AverageMetric
from crslab.evaluator.metrics.gen import PPLMetric
from crslab.system.base import BaseSystem
from crslab.system.utils.functions import ind2txt


class KBRDSystem(BaseSystem):
    """This is the system for KBRD model"""

    def __init__(self, opt, train_dataloader, valid_dataloader, test_dataloader, vocab, side_data, restore_system=False,
                 interact=False, debug=False, tensorboard=False):
        """

        Args:
            opt (dict): Indicating the hyper parameters.
            train_dataloader (BaseDataLoader): Indicating the train dataloader of corresponding dataset.
            valid_dataloader (BaseDataLoader): Indicating the valid dataloader of corresponding dataset.
            test_dataloader (BaseDataLoader): Indicating the test dataloader of corresponding dataset.
            vocab (dict): Indicating the vocabulary.
            side_data (dict): Indicating the side data.
            restore_system (bool, optional): Indicating if we store system after training. Defaults to False.
            interact (bool, optional): Indicating if we interact with system. Defaults to False.
            debug (bool, optional): Indicating if we train in debug mode. Defaults to False.
            tensorboard (bool, optional) Indicating if we monitor the training performance in tensorboard. Defaults to False. 

        """
        super(KBRDSystem, self).__init__(opt, train_dataloader, valid_dataloader, test_dataloader, vocab, side_data,
                                         restore_system, interact, debug, tensorboard)

        self.ind2tok = vocab['ind2tok']
        self.end_token_idx = vocab['end']
        self.item_ids = side_data['item_entity_ids']

        self.rec_optim_opt = opt['rec']
        self.conv_optim_opt = opt['conv']
        self.rec_epoch = self.rec_optim_opt['epoch']
        self.conv_epoch = self.conv_optim_opt['epoch']
        self.rec_batch_size = self.rec_optim_opt['batch_size']
        self.conv_batch_size = self.conv_optim_opt['batch_size']

        # wandb
        self.wandb = wandb.init(project="KBRD System", config=self.opt.opt)

    def rec_evaluate(self, rec_predict, item_label):
        rec_predict = rec_predict.cpu()
        rec_predict = rec_predict[:, self.item_ids]
        _, rec_ranks = torch.topk(rec_predict, 50, dim=-1)
        rec_ranks = rec_ranks.tolist()
        item_label = item_label.tolist()
        for rec_rank, label in zip(rec_ranks, item_label):
            label = self.item_ids.index(label)
            self.evaluator.rec_evaluate(rec_rank, label)

    def conv_evaluate(self, prediction, response):
        prediction = prediction.tolist()
        response = response.tolist()
        for p, r in zip(prediction, response):
            p_str = ind2txt(p, self.ind2tok, self.end_token_idx)
            r_str = ind2txt(r, self.ind2tok, self.end_token_idx)
            self.evaluator.gen_evaluate(p_str, [r_str])

    def step(self, batch, stage, mode):
        assert stage in ('rec', 'conv')
        assert mode in ('train', 'valid', 'test')

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)

        if stage == 'rec':
            rec_loss, rec_scores = self.model.forward(batch, mode, stage)
            rec_loss = rec_loss.sum()
            if mode == 'train':
                self.backward(rec_loss)
            else:
                self.rec_evaluate(rec_scores, batch['item'])
            rec_loss = rec_loss.item()
            self.evaluator.optim_metrics.add("rec_loss", AverageMetric(rec_loss))
        else:
            if mode != 'test':
                gen_loss, preds = self.model.forward(batch, mode, stage)
                if mode == 'train':
                    self.backward(gen_loss)
                else:
                    self.conv_evaluate(preds, batch['response'])
                gen_loss = gen_loss.item()
                self.evaluator.optim_metrics.add('gen_loss', AverageMetric(gen_loss))
                self.evaluator.gen_metrics.add("ppl", PPLMetric(gen_loss))
            else:
                preds = self.model.forward(batch, mode, stage)

                # 2023.02.02 cyf add
                for context, prediction, response in zip(batch['context_tokens'], preds, batch['response']):
                    pad_index = self.vocab["tok2ind"]["__pad__"]
                    index = 0
                    while index < len(context) - 1:
                        if context[index] != pad_index:
                            break
                        index += 1
                    c_str = ind2txt(context[index:], self.ind2tok, self.end_token_idx)

                    p_str = ind2txt(prediction[2:], self.ind2tok, self.end_token_idx)
                    r_str = ind2txt(response[1:], self.ind2tok, pad_index)
                    self.generation_result.append(["context:" + c_str, "prediction:" + p_str, "response:" + r_str])

                self.conv_evaluate(preds, batch['response'])

    def train_recommender(self):
        self.init_optim(self.rec_optim_opt, self.model.parameters())

        for epoch in range(self.rec_epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[Recommendation epoch {str(epoch)}]')
            logger.info('[Train]')
            for batch in self.train_dataloader.get_rec_data(self.rec_batch_size):
                self.step(batch, stage='rec', mode='train')
                self.wandb.log({
                "train_recloss": self.evaluator.optim_metrics["rec_loss"],
            })
            self.evaluator.report(epoch=epoch, mode='train')
            # val
            logger.info('[Valid]')
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.valid_dataloader.get_rec_data(self.rec_batch_size, shuffle=False):
                    self.step(batch, stage='rec', mode='valid')
                    self.wandb.log({
                    "valid_recloss": self.evaluator.optim_metrics['rec_loss'],
                })
                self.evaluator.report(epoch=epoch, mode='valid')
                self.wandb.log({
                    "hit@1": self.evaluator.rec_metrics["hit@1"],
                    "hit@10": self.evaluator.rec_metrics["hit@10"],
                    "hit@50": self.evaluator.rec_metrics["hit@50"],
                    "mrr@1": self.evaluator.rec_metrics["mrr@1"],
                    "mrr@10": self.evaluator.rec_metrics["mrr@10"],
                    "mrr@50": self.evaluator.rec_metrics["mrr@50"],
                    "ndcg@1": self.evaluator.rec_metrics["ndcg@1"],
                    "ndcg@10": self.evaluator.rec_metrics["ndcg@10"],
                    "ndcg@50": self.evaluator.rec_metrics["ndcg@50"],
                })
                # early stop
                metric = self.evaluator.optim_metrics['rec_loss']
                if self.early_stop(metric):
                    break
        # test
        logger.info('[Test]')
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.test_dataloader.get_rec_data(self.rec_batch_size, shuffle=False):
                self.step(batch, stage='rec', mode='test')
            self.evaluator.report(mode='test')

    def train_conversation(self):
        if os.environ["CUDA_VISIBLE_DEVICES"] == '-1':
            self.model.freeze_parameters()
        elif len(os.environ["CUDA_VISIBLE_DEVICES"]) == 1:
            self.model.freeze_parameters()
        else:
            self.model.module.freeze_parameters()
        self.init_optim(self.conv_optim_opt, self.model.parameters())

        for epoch in range(self.conv_epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[Conversation epoch {str(epoch)}]')
            logger.info('[Train]')
            for batch in self.train_dataloader.get_conv_data(batch_size=self.conv_batch_size):
                self.step(batch, stage='conv', mode='train')
                self.wandb.log({
                "train_genloss": self.evaluator.optim_metrics["gen_loss"],
                "train_ppl": self.evaluator.gen_metrics["ppl"],
            })
            self.evaluator.report(epoch=epoch, mode='train')
            # val
            logger.info('[Valid]')
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.valid_dataloader.get_conv_data(batch_size=self.conv_batch_size, shuffle=False):
                    self.step(batch, stage='conv', mode='valid')
                    self.wandb.log({
                    "valid_genloss": self.evaluator.optim_metrics['gen_loss'],
                    "valid_ppl": self.evaluator.gen_metrics['ppl'],
                })
                self.evaluator.report(epoch=epoch, mode='valid')
                self.wandb.log({
                    "bleu@1": self.evaluator.gen_metrics['bleu@1'],
                    "bleu@2": self.evaluator.gen_metrics['bleu@2'],
                    "bleu@3": self.evaluator.gen_metrics['bleu@3'],
                    "bleu@4": self.evaluator.gen_metrics['bleu@4'],
                    "dist@1": self.evaluator.gen_metrics['dist@1'],
                    "dist@2": self.evaluator.gen_metrics['dist@2'],
                    "dist@3": self.evaluator.gen_metrics['dist@3'],
                    "dist@4": self.evaluator.gen_metrics['dist@4'],
                })
                # early stop
                metric = self.evaluator.optim_metrics['gen_loss']
                if self.early_stop(metric):
                    break
        # test
        logger.info('[Test]')
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.test_dataloader.get_conv_data(batch_size=self.conv_batch_size, shuffle=False):
                self.step(batch, stage='conv', mode='test')

            # 2023.02.02 cyf add
            f = open("./gen_result/" + self.opt["dataset"] + '_' + self.opt["model_name"] + "_" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + ".txt", "w", encoding="UTF-8")
            f.write("\n\n".join(["\n".join(samples) for samples in self.generation_result]))
            f.close()

            self.evaluator.report(mode='test')

    def fit(self):
        self.train_recommender()
        self.train_conversation()

    def interact(self):
        pass
