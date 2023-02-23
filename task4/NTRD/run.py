#!/usr/bin/env python3
import random
from random import sample
import time
import os
import cProfile

# Copyright (c) Microsoft Coporation. and its affiliates.
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

# TODO List:
# * More logging (e.g. to files), make things prettier.

import numpy as np
from torch.utils.data import RandomSampler
from tqdm import tqdm
from math import exp
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import signal
import json
import argparse
import pickle as pkl
from dataset import dataset, CRSdataset
from model import CrossModel
import torch.nn as nn
from torch import optim
import torch
import wandb

try:
    import torch.version
    import torch.distributed as dist

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
from nltk.translate.bleu_score import sentence_bleu

MOVIE_TOKEN_INDEX = 6


def is_distributed():
    """
    Returns True if we are in distributed mode.
    """
    return TORCH_AVAILABLE and dist.is_available() and dist.is_initialized()


def setup_args():
    train = argparse.ArgumentParser()
    train.add_argument("-max_c_length", "--max_c_length", type=int, default=256)
    train.add_argument("-max_r_length", "--max_r_length", type=int, default=30)
    train.add_argument("-beam", "--beam", type=int, default=1)
    # train.add_argument("-max_r_length","--max_r_length",type=int,default=256)
    train.add_argument("-batch_size", "--batch_size", type=int, default=32)
    train.add_argument("-max_count", "--max_count", type=int, default=5)
    train.add_argument("-use_cuda", "--use_cuda", type=bool, default=True)
    train.add_argument("-is_template", "--is_template", type=bool, default=True)
    train.add_argument("-infomax_pretrain", "--infomax_pretrain", type=bool, default=True)
    train.add_argument("-load_dict", "--load_dict", type=str, default=None)
    train.add_argument("-learningrate", "--learningrate", type=float, default=1e-3)
    train.add_argument("-optimizer", "--optimizer", type=str, default='adam')
    train.add_argument("-momentum", "--momentum", type=float, default=0)
    train.add_argument("-is_finetune", "--is_finetune", type=bool, default=True)
    train.add_argument("-embedding_type", "--embedding_type", type=str, default='random')
    train.add_argument("-save_exp_name", "--save_exp_name", type=str, default='saved_model')
    # train.add_argument("-saved_hypo_txt","--saved_hypo_txt",type=str,default='case_file/output_hypo_latest.txt')
    train.add_argument("-saved_hypo_txt", "--saved_hypo_txt", type=str, default=None)
    train.add_argument("-load_model_pth", "--load_model_pth", type=str, default='saved_model/best_dist4.pkl')
    train.add_argument("-epoch", "--epoch", type=int, default=30)
    train.add_argument("-gpu", "--gpu", type=str, default='0')
    train.add_argument("-gradient_clip", "--gradient_clip", type=float, default=0.1)
    train.add_argument("-gen_loss_weight", "--gen_loss_weight", type=float, default=5)
    train.add_argument("-embedding_size", "--embedding_size", type=int, default=300)

    train.add_argument("-n_heads", "--n_heads", type=int, default=2)
    train.add_argument("-n_layers", "--n_layers", type=int, default=2)
    train.add_argument("-ffn_size", "--ffn_size", type=int, default=300)

    train.add_argument("-dropout", "--dropout", type=float, default=0.1)
    train.add_argument("-attention_dropout", "--attention_dropout", type=float, default=0.0)
    train.add_argument("-relu_dropout", "--relu_dropout", type=float, default=0.1)

    train.add_argument("-learn_positional_embeddings", "--learn_positional_embeddings", type=bool, default=True)
    train.add_argument("-embeddings_scale", "--embeddings_scale", type=bool, default=True)

    train.add_argument("-n_movies", "--n_movies", type=int, default=68956)
    train.add_argument("-n_entity", "--n_entity", type=int, default=97285)
    train.add_argument("-n_relation", "--n_relation", type=int, default=214)
    train.add_argument("-n_concept", "--n_concept", type=int, default=898190)
    train.add_argument("-n_con_relation", "--n_con_relation", type=int, default=48)
    train.add_argument("-dim", "--dim", type=int, default=128)
    train.add_argument("-n_hop", "--n_hop", type=int, default=2)
    train.add_argument("-kge_weight", "--kge_weight", type=float, default=1)
    train.add_argument("-l2_weight", "--l2_weight", type=float, default=2.5e-6)
    train.add_argument("-n_memory", "--n_memory", type=float, default=32)
    train.add_argument("-item_update_mode", "--item_update_mode", type=str, default='0,1')
    train.add_argument("-using_all_hops", "--using_all_hops", type=bool, default=True)
    train.add_argument("-num_bases", "--num_bases", type=int, default=8)

    return train

class TrainLoop_fusion_gen():
    def __init__(self, opt, is_finetune):
        self.opt = opt
        self.train_dataset = dataset('data/train_data.jsonl', opt)

        self.dict = self.train_dataset.word2index
        self.index2word = {self.dict[key]: key for key in self.dict}

        self.movieID2selection_label = pkl.load(open('data/movieID2selection_label.pkl', 'rb'))
        self.selection_label2movieID = {self.movieID2selection_label[key]: key for key in self.movieID2selection_label}
        self.id2entity = pkl.load(open('data/id2entity.pkl', 'rb'))

        # self.total_novel_movies = TOTAL_NOVEL_MOVIES

        self.batch_size = self.opt['batch_size']
        self.epoch = self.opt['epoch']

        self.use_cuda = opt['use_cuda']
        if opt['load_dict'] != None:
            self.load_data = True
        else:
            self.load_data = False
        self.is_finetune = False

        self.is_template = opt['is_template']

        self.movie_ids = pkl.load(open("data/movie_ids.pkl", "rb"))
        # Note: we cannot change the type of metrics ahead of time, so you
        # should correctly initialize to floats or ints here

        self.metrics_rec = {"recall@1": 0, "recall@10": 0, "recall@50": 0, "loss": 0, "count": 0}
        self.metrics_gen = {"dist1": 0, "dist2": 0, "dist3": 0, "dist4": 0, "bleu1": 0, "bleu2": 0, "bleu3": 0,
                            "bleu4": 0, "count": 0, "true_recall_movie_count": 0, "res_movie_recall": 0.0,
                            "recall@1": 0, "recall@10": 0, "recall@50": 0}

        self.build_model(is_finetune=True)

        if opt['load_dict'] is not None:
            # load model parameters if available
            print('[ Loading existing model params from {} ]'
                  ''.format(opt['load_dict']))
            states = self.model.load(opt['load_dict'])
        else:
            states = {}

        self.init_optim(
            [p for p in self.model.parameters() if p.requires_grad],
            optim_states=states.get('optimizer'),
            saved_optim_type=states.get('optimizer_type')
        )

    def build_model(self, is_finetune):
        self.model = CrossModel(self.opt, self.dict, is_finetune)
        if self.opt['embedding_type'] != 'random':
            pass
        if self.use_cuda:
            self.model.cuda()

    def train(self):
        # self.model.load_model()
        losses = []
        best_val_gen = 0
        best_val_rec = 0
        gen_stop = False

        for i in range(self.epoch * 1):
            train_set = CRSdataset(self.train_dataset.data_process(True), self.opt['n_entity'], self.opt['n_concept'])
            train_dataset_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                               batch_size=self.batch_size,
                                                               shuffle=True)
            num = 0
            for context, c_lengths, response, r_length, mask_response, mask_r_length, entity, entity_vector, movie, concept_mask, dbpedia_mask, concept_vec, db_vec, rec, movies_gth, movie_nums in tqdm(train_dataset_loader):
                seed_sets = []
                batch_size = context.shape[0]
                for b in range(batch_size):
                    seed_set = entity[b].nonzero().view(-1).tolist()
                    seed_sets.append(seed_set)
                self.model.train()
                self.zero_grad()

                scores, preds, rec_scores, rec_loss, gen_loss, mask_loss, info_db_loss, info_con_loss, selection_loss, matching_pred, matching_scores = self.model(
                    context.cuda(), response.cuda(), mask_response.cuda(), concept_mask, dbpedia_mask, seed_sets, movie,
                    concept_vec, db_vec, entity_vector.cuda(), rec, movies_gth.cuda(), movie_nums, test=False)

                gen_loss = self.opt['gen_loss_weight'] * gen_loss
                joint_loss = gen_loss + selection_loss
                wandb.log({'train_gen_loss' : gen_loss, 'train_rec_loss': selection_loss})
                losses.append([gen_loss, selection_loss])
                self.backward(joint_loss)
                self.update_params()
                if num % 100 == 0:
                    print('gen loss is %f' % (sum([l[0] for l in losses]) / len(losses)))
                    print('selection_loss is %f' % (sum([l[1] for l in losses]) / len(losses)))
                    losses = []
                num += 1

            output_metrics_gen = self.val()
            if best_val_gen > output_metrics_gen["dist4"]:
                pass
            else:
                best_val_gen = output_metrics_gen["dist4"]
                self.model.save_model(model_path=self.opt['save_exp_name'], model_name='best_dist4.pkl')
                print("Best Dist4 generator model saved once------------------------------------------------")
            print("best dist4 is :", best_val_gen)

            # if best_val_rec > output_metrics_gen["recall@50"] + output_metrics_gen["recall@1"]:
            #     pass
            # else:
            #     best_val_rec = output_metrics_gen["recall@50"] + output_metrics_gen["recall@1"]
            #     self.model.save_model(model_path=self.opt['save_exp_name'], model_name='best_Rec.pkl')
            #     print("Best Recall generator model saved once------------------------------------------------")
            # print("best res_movie_R@1 is :", output_metrics_gen["recall@1"])
            # print("best res_movie_R@10 is :", output_metrics_gen["recall@10"])
            # print("best res_movie_R@50 is :", output_metrics_gen["recall@50"])
            # if losses.__len__() != 0:
            #     print('cur selection_loss is %f' % (sum([l[1] for l in losses]) / len(losses)))
            print('cur Epoch is : ', i)

            # if i % 5 ==0: # save each 5 epoch
            #     model_name = self.opt['save_exp_name'] + '_' + str(i) + '.pkl'
            #     self.model.save_model(model_name=model_name)
            #     print("generator model saved once------------------------------------------------")
            #     print('cur selection_loss is %f'%(sum([l[1] for l in losses])/len(losses)))

        _ = self.val(is_test=True, last_epoch=True)

    def val(self, is_test=False, last_epoch=False):
        self.metrics_gen = {"ppl": 0, "dist1": 0, "dist2": 0, "dist3": 0, "dist4": 0, "bleu1": 0, "bleu2": 0,
                            "bleu3": 0, "bleu4": 0, "count": 0, "true_recall_movie_count": 0, "res_movie_recall": 0.0,
                            "recall@1": 0, "recall@10": 0, "recall@50": 0}
        self.metrics_rec = {"recall@1": 0, "recall@10": 0, "recall@50": 0, "loss": 0, "gate": 0, "count": 0,
                            'gate_count': 0}
        self.model.eval()
        if is_test:
            val_dataset = dataset('data/test_data.jsonl', self.opt)
        else:
            val_dataset = dataset('data/valid_data.jsonl', self.opt)
        val_set = CRSdataset(val_dataset.data_process(True), self.opt['n_entity'], self.opt['n_concept'])
        val_dataset_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=self.batch_size, shuffle=False)
        inference_sum = []
        golden_sum = []
        context_sum = []
        losses = []
        recs = []

        for context, c_lengths, response, r_length, mask_response, mask_r_length, entity, entity_vector, movie, concept_mask, dbpedia_mask, concept_vec, db_vec, rec, movies_gth, movie_nums in tqdm(val_dataset_loader):
            with torch.no_grad():
                seed_sets = []
                batch_size = context.shape[0]
                for b in range(batch_size):
                    seed_set = entity[b].nonzero().view(-1).tolist()
                    seed_sets.append(seed_set)

                # -----dump , run the first time only to get the gen_loss, could be optimized here ------By Jokie 2021/04/15
                _, _, _, _, gen_loss, mask_loss, info_db_loss, info_con_loss, selection_loss, _, _ = self.model(
                    context.cuda(), response.cuda(), mask_response.cuda(), concept_mask, dbpedia_mask, seed_sets, movie,
                    concept_vec, db_vec, entity_vector.cuda(), rec, movies_gth.cuda(), movie_nums, test=False)
                scores, preds, rec_scores, rec_loss, _, mask_loss, info_db_loss, info_con_loss, selection_loss, matching_pred, matching_scores = self.model(
                    context.cuda(), response.cuda(), mask_response.cuda(), concept_mask, dbpedia_mask, seed_sets, movie,
                    concept_vec, db_vec, entity_vector.cuda(), rec, movies_gth.cuda(), movie_nums, test=True, maxlen=20,
                    bsz=batch_size)

                # golden_sum.extend(self.vector2sentence(response.cpu()))
                # inference_sum.extend(self.vector2sentence(preds.cpu()))
                # context_sum.extend(self.vector2sentence(context.cpu()))

                wandb.log({
                    'valid_gen_loss' : gen_loss,
                    'valid_rec_loss' : selection_loss
                })

                self.all_response_movie_recall_cal(preds.cpu(), matching_scores.cpu(), movies_gth.cpu())

            # -----------template pro-process gth response and prediction--------------------
            if self.is_template:
                golden_sum.extend(self.template_vector2sentence(response.cpu(), movies_gth.cpu()))
                if matching_pred is not None:
                    inference_sum.extend(self.template_vector2sentence(preds.cpu(), matching_pred.cpu()))
                else:
                    inference_sum.extend(self.template_vector2sentence(preds.cpu(), None))
            else:
                golden_sum.extend(self.vector2sentence(response.cpu()))
                inference_sum.extend(self.vector2sentence(preds.cpu()))
            context_sum.extend(self.vector2sentence(context.cpu()))

            recs.extend(rec.cpu())
            losses.append(torch.mean(gen_loss))

        if last_epoch:
            folder_name = "./cases/" + time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime()) +"/"
            os.mkdir(folder_name)
            f = open(folder_name +'context_test.txt', 'w', encoding='utf-8')
            f.writelines([' '.join(sen) + '\n' for sen in context_sum])
            f.close()

            f = open(folder_name +'output_self_attn_no_decode_first_filled_template_test.txt', 'w', encoding='utf-8')
            f.writelines([' '.join(sen) + '\n' for sen in inference_sum])
            f.close()

            f = open(folder_name +'golden_test.txt', 'w', encoding='utf-8')
            f.writelines([' '.join(sen) + '\n' for sen in golden_sum])
            f.close()

            f = open(folder_name +'case_visualize.txt', 'w', encoding='utf-8')
            for cont, hypo, gold in zip(context_sum, inference_sum, golden_sum):
                f.writelines('context: ' + ' '.join(cont) + '\n')
                f.writelines('hypo: ' + ' '.join(hypo) + '\n')
                f.writelines('gold: ' + ' '.join(gold) + '\n')
                f.writelines('\n')
            f.close()

        self.metrics_cal_gen(losses, inference_sum, golden_sum, recs, beam=self.opt['beam'])

        output_dict_gen = {}
        for key in self.metrics_gen:
            if 'bleu' in key:
                output_dict_gen[key] = self.metrics_gen[key] / self.metrics_gen['count']
            else:
                output_dict_gen[key] = self.metrics_gen[key]
        print(output_dict_gen)
        wandb.log({
            'test_ppl':output_dict_gen['ppl'],
            'bleu1':output_dict_gen['bleu1'],
            'bleu2':output_dict_gen['bleu2'],
            'bleu3':output_dict_gen['bleu3'],
            'bleu4':output_dict_gen['bleu4'],
            'dist1':output_dict_gen['dist1'],
            'dist2':output_dict_gen['dist2'],
            'dist3':output_dict_gen['dist3'],
            'dist4':output_dict_gen['dist4'],
        })

        # default None and always set None
        if self.opt['saved_hypo_txt'] is not None:
            f = open(self.opt['saved_hypo_txt'], 'w', encoding='utf-8')
            f.writelines([' '.join(sen) + '\n' for sen in inference_sum])
            f.close()

        return output_dict_gen

    def all_response_movie_recall_cal(self, decode_preds, matching_scores, labels):

        # matching_scores is non-mask version [bsz, seq_len, matching_vocab]
        # decode_preds [bsz, seq_len]
        # labels [bsz, movie_length_with_padding]
        # print('decode_preds shape', decode_preds.shape)
        # print('matching_scores shape', matching_scores.shape)
        # print('labels shape', labels.shape)
        
        return
        decode_preds = decode_preds[:, 1:]  # removing the start index

        labels = labels * (labels != -1)  # removing the padding token

        batch_size, seq_len = decode_preds.shape[0], decode_preds.shape[1]
        for cur_b in range(batch_size):
            for cur_seq_len in range(seq_len):
                if decode_preds[cur_b][cur_seq_len] == MOVIE_TOKEN_INDEX:  # word id is 15
                    _, pred_idx = torch.topk(matching_scores[cur_b][cur_seq_len], k=100, dim=-1)
                    targets = labels[cur_b]
                    for target in targets:
                        self.metrics_gen["recall@1"] += int(target in pred_idx[:1].tolist())
                        self.metrics_gen["recall@10"] += int(target in pred_idx[:10].tolist())
                        self.metrics_gen["recall@50"] += int(target in pred_idx[:50].tolist())


    def metrics_cal_gen(self, rec_loss, preds, responses, recs, beam=1):
        def bleu_cal(sen1, tar1):
            bleu1 = sentence_bleu([tar1], sen1, weights=(1, 0, 0, 0))
            bleu2 = sentence_bleu([tar1], sen1, weights=(0, 1, 0, 0))
            bleu3 = sentence_bleu([tar1], sen1, weights=(0, 0, 1, 0))
            bleu4 = sentence_bleu([tar1], sen1, weights=(0, 0, 0, 1))
            return bleu1, bleu2, bleu3, bleu4

        def response_movie_recall_cal(sen1, tar1):
            for word in sen1:
                if '@' in word:  # if is movie
                    if word in tar1:  # if in gth
                        return int(1)
                    else:
                        return int(0)
            return int(0)

        def distinct_metrics(outs):
            # outputs is a list which contains several sentences, each sentence contains several words
            unigram_count = 0
            bigram_count = 0
            trigram_count = 0
            quagram_count = 0
            unigram_set = set()
            bigram_set = set()
            trigram_set = set()
            quagram_set = set()
            for sen in outs:
                for word in sen:
                    unigram_count += 1
                    unigram_set.add(word)
                for start in range(len(sen) - 1):
                    bg = str(sen[start]) + ' ' + str(sen[start + 1])
                    bigram_count += 1
                    bigram_set.add(bg)
                for start in range(len(sen) - 2):
                    trg = str(sen[start]) + ' ' + str(sen[start + 1]) + ' ' + str(sen[start + 2])
                    trigram_count += 1
                    trigram_set.add(trg)
                for start in range(len(sen) - 3):
                    quag = str(sen[start]) + ' ' + str(sen[start + 1]) + ' ' + str(sen[start + 2]) + ' ' + str(
                        sen[start + 3])
                    quagram_count += 1
                    quagram_set.add(quag)
            # dis1 = len(unigram_set) / len(outs)  # unigram_count
            # dis2 = len(bigram_set) / len(outs)  # bigram_count
            # dis3 = len(trigram_set) / len(outs)  # trigram_count
            # dis4 = len(quagram_set) / len(outs)  # quagram_count
            dis1 = len(unigram_set) / unigram_count
            dis2 = len(bigram_set) / bigram_count
            dis3 = len(trigram_set) / trigram_count
            dis4 = len(quagram_set) / quagram_count
            return dis1, dis2, dis3, dis4

        predict_s = preds
        golden_s = responses
        # print(rec_loss[0])
        self.metrics_gen["ppl"] += sum([exp(ppl) for ppl in rec_loss]) / len(rec_loss)
        generated = []
        total_movie_gth_response_cnt = 0
        have_movie_res_cnt = 0
        loop = 0
        total_item_response_cnt = 0
        total_hypo_word_count = 0
        novel_pred_movies = []
        non_novel_pred_movies = []
        # for out, tar, rec in zip(predict_s, golden_s, recs):
        for out in predict_s:
            tar = golden_s[loop // beam]
            loop = loop + 1
            bleu1, bleu2, bleu3, bleu4 = bleu_cal(out, tar)
            generated.append(out)
            self.metrics_gen['bleu1'] += bleu1
            self.metrics_gen['bleu2'] += bleu2
            self.metrics_gen['bleu3'] += bleu3
            self.metrics_gen['bleu4'] += bleu4
            self.metrics_gen['count'] += 1
            self.metrics_gen['true_recall_movie_count'] += response_movie_recall_cal(out, tar)
            for word in out:
                total_hypo_word_count += 1
                if '@' in word:
                    total_item_response_cnt += 1
                    try:
                        int_movie_id = int(word[1:])
                        if int_movie_id in set(self.total_novel_movies):
                            novel_pred_movies.append(int_movie_id)
                        else:
                            non_novel_pred_movies.append(int_movie_id)
                    except:
                        non_novel_pred_movies.append(word[1:])
                        pass

        total_target_word_count = 0
        for tar in golden_s:
            for word in tar:
                total_target_word_count += 1
                if '@' in word:
                    total_movie_gth_response_cnt += 1
            for word in tar:
                if '@' in word:
                    have_movie_res_cnt += 1
                    break

        dis1, dis2, dis3, dis4 = distinct_metrics(generated)
        self.metrics_gen['dist1'] = dis1
        self.metrics_gen['dist2'] = dis2
        self.metrics_gen['dist3'] = dis3
        self.metrics_gen['dist4'] = dis4

        self.metrics_gen['res_movie_recall'] = self.metrics_gen['true_recall_movie_count'] / have_movie_res_cnt
        self.metrics_gen["recall@1"] = self.metrics_gen["recall@1"] / have_movie_res_cnt
        self.metrics_gen["recall@10"] = self.metrics_gen["recall@10"] / have_movie_res_cnt
        self.metrics_gen["recall@50"] = self.metrics_gen["recall@50"] / have_movie_res_cnt
        print('----------' * 10)
        print('total_movie_gth_response_cnt: ', total_movie_gth_response_cnt)
        print('total_gth_response_cnt: ', len(golden_s))
        print('total_hypo_response_cnt: ', len(predict_s))
        print('hypo item ratio: ', total_item_response_cnt / len(predict_s))
        print('target item ratio: ', total_movie_gth_response_cnt / len(golden_s))
        print('have_movie_res_cnt: ', have_movie_res_cnt)
        print('len(novel_pred_movies): ', len(novel_pred_movies))
        print('novel_pred_movies: ', novel_pred_movies)
        print('len(non_novel_pred_movies): ', len(non_novel_pred_movies))
        print('num of different predicted movies: ', len(set(non_novel_pred_movies)))
        print('non_novel_pred_movies: ', set(non_novel_pred_movies))
        print('----------' * 10)

    def vector2sentence(self, batch_sen):
        sentences = []
        for sen in batch_sen.numpy().tolist():
            sentence = []
            for word in sen:
                if word > 3:
                    sentence.append(self.index2word[word])
                    # if word==MOVIE_TOKEN_INDEX: #if MOVIE token
                    #     sentence.append(self.selection_label2movieID[selection_label])
                elif word == 3:
                    sentence.append('_UNK_')
            sentences.append(sentence)
        return sentences

    def template_vector2sentence(self, batch_sen, batch_selection_pred):
        sentences = []
        all_movie_labels = []
        if batch_selection_pred is not None:
            # batch_selection_pred = batch_selection_pred * (batch_selection_pred != -1)
            batch_selection_pred = torch.masked_select(batch_selection_pred, (batch_selection_pred != -1))
            for movie in batch_selection_pred.numpy().tolist():
                all_movie_labels.append(movie)

        # print('all_movie_labels:', all_movie_labels)
        curr_movie_token = 0
        for sen in batch_sen.numpy().tolist():
            sentence = []
            for word in sen:
                if word > 3:
                    if word == MOVIE_TOKEN_INDEX:  # if MOVIE token
                        # print('all_movie_labels[curr_movie_token]',all_movie_labels[curr_movie_token])
                        # print('selection_label2movieID',self.selection_label2movieID[all_movie_labels[curr_movie_token]])

                        # WAY1: original method
                        sentence.append('@' + str(self.selection_label2movieID[all_movie_labels[curr_movie_token]]))

                        # WAY2: print out the movie name, but should comment when calculating the gen metrics
                        # if self.id2entity[self.selection_label2movieID[all_movie_labels[curr_movie_token]]] is not None:
                        #     sentence.append(self.id2entity[self.selection_label2movieID[all_movie_labels[curr_movie_token]]].split('/')[-1])
                        # else:
                        #     sentence.append('@' + str(self.selection_label2movieID[all_movie_labels[curr_movie_token]]))

                        curr_movie_token += 1
                    else:
                        sentence.append(self.index2word[word])

                elif word == 3:
                    sentence.append('_UNK_')
            sentences.append(sentence)

            # print('[DEBUG]sentence : ')
            # print(u' '.join(sentence).encode('utf-8').strip())

        assert curr_movie_token == len(all_movie_labels)
        return sentences

    @classmethod
    def optim_opts(self):
        """
        Fetch optimizer selection.

        By default, collects everything in torch.optim, as well as importing:
        - qhm / qhmadam if installed from github.com/facebookresearch/qhoptim

        Override this (and probably call super()) to add your own optimizers.
        """
        # first pull torch.optim in
        optims = {k.lower(): v for k, v in optim.__dict__.items()
                  if not k.startswith('__') and k[0].isupper()}
        try:
            import apex.optimizers.fused_adam as fused_adam
            optims['fused_adam'] = fused_adam.FusedAdam
        except ImportError:
            pass

        try:
            # https://openreview.net/pdf?id=S1fUpoR5FQ
            from qhoptim.pyt import QHM, QHAdam
            optims['qhm'] = QHM
            optims['qhadam'] = QHAdam
        except ImportError:
            # no QHM installed
            pass

        return optims

    def init_optim(self, params, optim_states=None, saved_optim_type=None):
        """
        Initialize optimizer with model parameters.

        :param params:
            parameters from the model

        :param optim_states:
            optional argument providing states of optimizer to load

        :param saved_optim_type:
            type of optimizer being loaded, if changed will skip loading
            optimizer states
        """

        opt = self.opt

        # set up optimizer args
        lr = opt['learningrate']
        kwargs = {'lr': lr}
        kwargs['amsgrad'] = True
        kwargs['betas'] = (0.9, 0.999)

        optim_class = self.optim_opts()[opt['optimizer']]
        self.optimizer = optim_class(params, **kwargs)

    def backward(self, loss):
        """
        Perform a backward pass. It is recommended you use this instead of
        loss.backward(), for integration with distributed training and FP16
        training.
        """
        loss.backward()

    def update_params(self):
        """
        Perform step of optimization, clipping gradients and adjusting LR
        schedule if needed. Gradient accumulation is also performed if agent
        is called with --update-freq.

        It is recommended (but not forced) that you call this in train_step.
        """
        update_freq = 1
        if update_freq > 1:
            # we're doing gradient accumulation, so we don't only want to step
            # every N updates instead
            self._number_grad_accum = (self._number_grad_accum + 1) % update_freq
            if self._number_grad_accum != 0:
                return

        if self.opt['gradient_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.opt['gradient_clip']
            )

        self.optimizer.step()

    def zero_grad(self):
        """
        Zero out optimizer.

        It is recommended you call this in train_step. It automatically handles
        gradient accumulation if agent is called with --update-freq.
        """
        self.optimizer.zero_grad()


if __name__ == '__main__':
    args = setup_args().parse_args()
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    print('CUDA_VISIBLE_DEVICES:', os.environ['CUDA_VISIBLE_DEVICES'])
    print(vars(args))
    wandb.init(project='EOD-T4-NTRD', config=args)
    if args.is_finetune == False:
        pass
        # loop = TrainLoop_fusion_rec(vars(args), is_finetune=False)
        # loop.model.load_model('saved_model/net_parameter1_bu.pkl')
        # loop.train()
    else:
        loop = TrainLoop_fusion_gen(vars(args), is_finetune=True)
        # Tips: should at least load one of the model By Jokie

        # if validation
        # WAY1:
        # loop.model.load_model('saved_model/matching_linear_model/generation_model_best.pkl')

        # WAY2:
        # loop.model.load_model('saved_model/sattn_dialog_model_best.pkl')
        # loop.model.load_model('saved_model/generation_model_best.pkl')
        # loop.model.load_model('saved_model/generation_model.pkl')
        # loop.model.load_model('saved_model/self_attn_generation_model_22.pkl')

        # WAY3: insert
        # loop.model.load_model()

        # 这一行注释掉了cyf 2022.11.19
        # loop.model.load_model(args.load_model_pth)

        loop.train()
