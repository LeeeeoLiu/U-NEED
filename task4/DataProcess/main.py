import json
import pickle

NTRD_STYLE = "NTRD"
CRSLab_STYLE = "CRSLab"
GPT2_STYLE = "GPT2"
domain_list = ["美妆行业", "手机行业", "服装行业", "鞋类行业", "大家电行业"]


class DataProcess:
    def __init__(self, style):
        self.item_id_dict = {}  # 因为有重复id, 且要有序, 因此采用dict结构
        self.entity_name_index_dict = {}
        self.relation_name_index_dict = {"客服id": 0}
        self.dialogue_data = {domain: [] for domain in domain_list}
        self.sen_tokenize_sen_dict = {}
        self.kg_item = {}
        self.text_dict = {}
        self.style = style

        self.load_tokenize_sentence()
        self.load_dialogue_data()
        self.load_item_kg()

        if style == NTRD_STYLE:
            self.dump_data_ntrd()
        elif style == CRSLab_STYLE:
            self.dump_data_crslab()
        elif style == GPT2_STYLE:
            self.dump_data_gpt2()

    # utils
    def entity_name2index(self, name):
        if name not in self.entity_name_index_dict.keys():
            self.entity_name_index_dict[name] = self.entity_name_index_dict.__len__()
        return self.entity_name_index_dict[name]

    def relation_name2index(self, name):
        if name not in self.relation_name_index_dict.keys():
            self.relation_name_index_dict[name] = self.relation_name_index_dict.__len__()
        return self.relation_name_index_dict[name]

    def item_id2index(self, item_id):
        if item_id not in self.item_id_dict.keys():
            self.item_id_dict[item_id] = self.item_id_dict.__len__()
        return self.item_id_dict[item_id]

    # ----------------------------------------------------------------------------------------------------------
    # load data and process and stored in self
    def load_tokenize_sentence(self):
        if self.style == CRSLab_STYLE:
            self.sen_tokenize_sen_dict = pickle.load(open("ltp_vocab_list/sen_tokenize_sen_dict.pkl", "rb"))
        elif self.style == GPT2_STYLE:
            self.sen_tokenize_sen_dict = pickle.load(open("gpt2_vocab_list/sen_tokenize_sen_dict.pkl", "rb"))

    def load_dialogue_data(self):
        dialogue_data = json.load(open("std_ali_data/air_dataset_0210_encrypt.json", "r"))

        session_index = 0
        for session in dialogue_data:
            session_id = session["sid"]

            message_list = []
            temp_item_id_list = []

            for utterance in session["dialogue"]:
                # 调整对话语句内容, 收集商品信息
                if utterance["rec_item_id"].__len__() != 0:
                    temp_item_id_list.extend(utterance["rec_item_id"])
                    temp_str = ""
                    for item_id in utterance["rec_item_id"]:
                        self.entity_name2index(item_id)
                        self.item_id2index(item_id)
                        temp_str += "@" + item_id + " "

                    if utterance["send_content"] == "仅发送商品链接":
                        utterance["send_content"] = temp_str[:-1]
                    else:
                        utterance["send_content"] = temp_str + utterance["send_content"]

                # 收集entity信息
                temp_entity_set = set()
                for attribute in utterance["attributes"]:
                    if attribute["key"] != "":
                        self.entity_name2index(attribute["key"])
                        temp_entity_set.add(attribute["key"])
                    if attribute["value"] != "":
                        self.entity_name2index(attribute["value"])
                        temp_entity_set.add(attribute["value"])

                self.text_dict[utterance["send_content"]] = list(temp_entity_set)

                # 得到符合格式要求的数据
                if self.style == NTRD_STYLE:
                    message_list.append({
                        "timeOffset": utterance["seq_no"],
                        "text": utterance["send_content"],
                        "senderWorkerId": session["userid"] if utterance["sender_type"] == "用户" else session["sellerid"],
                        "messageId": session_index * 100 + utterance["seq_no"]
                    })
                elif self.style == CRSLab_STYLE or self.style == GPT2_STYLE:
                    message_list.append({
                        "uid": utterance["seq_no"] - 1,
                        "role": "Seeker" if utterance["sender_type"] == "用户" else "Recommender",
                        "text": self.sen_tokenize_sen_dict[utterance["send_content"]],
                        "movies": utterance["rec_item_id"],
                        "entity": list(temp_entity_set),
                        "word": []
                    })

            session_index += 1

            if self.style == NTRD_STYLE:
                self.dialogue_data[session["domain"]].append({
                    "movieMentions": {item_id: item_id for item_id in temp_item_id_list},
                    "respondentQuestions": {item_id: {"suggested": 0, "seen": 1, "liked": 1} for item_id in temp_item_id_list},
                    "messages": message_list,
                    "conversationId": session_id,
                    "respondentWorkerId": session["sellerid"],
                    "initiatorWorkerId": session["userid"],
                    "initiatorQuestions": {item_id: {"suggested": 0, "seen": 0, "liked": 1 if item_id == temp_item_id_list[-1] else 0} for item_id in temp_item_id_list}
                })
            elif self.style == CRSLab_STYLE or self.style == GPT2_STYLE:
                self.dialogue_data[session["domain"]].append({
                    "conv_id": session_id,
                    "dialog": message_list,
                })

    def load_item_kg(self):
        file = open("std_ali_data/kg_item_air_dataset_with_encrypt_0210.txt", "r")
        lines = file.readlines()
        file.close()

        for line in lines[1:]:
            item_id, seller_id, attribute, value = line.split(",")
            seller_id = seller_id[:-1]

            if not self.kg_item.keys().__contains__(self.entity_name2index(item_id)):
                self.kg_item[self.entity_name2index(item_id)] = [(0, self.entity_name2index(seller_id))]
                self.item_id2index(item_id)

            self.kg_item[self.entity_name2index(item_id)].append((self.relation_name2index(attribute), self.entity_name2index(value)))

    # ------------------------------------------------------------------------------------------
    # dump data into files
    def dump_dialogue_data(self, folder_name, single_domain_dialogue_data):
        if self.style == NTRD_STYLE:
            full_file = open("ali_ntrd_result/" + folder_name + "/full_data.jsonl", "w", encoding="UTF-8")
            train_file = open("ali_ntrd_result/" + folder_name + "/train_data.jsonl", "w", encoding="UTF-8")
            valid_file = open("ali_ntrd_result/" + folder_name + "/valid_data.jsonl", "w", encoding="UTF-8")
            test_file = open("ali_ntrd_result/" + folder_name + "/test_data.jsonl", "w", encoding="UTF-8")

            json_dialogue_data = [json.dumps(i, ensure_ascii=False) for i in single_domain_dialogue_data]
            num = len(json_dialogue_data) / 10
            full_file.write("\n".join(json_dialogue_data) + "\n")
            train_file.write("\n".join(json_dialogue_data[:int(num * 8)]) + "\n")
            valid_file.write("\n".join(json_dialogue_data[int(num * 8):int(num * 9)]) + "\n")
            test_file.write("\n".join(json_dialogue_data[int(num * 9):]) + "\n")

            return single_domain_dialogue_data[:int(num * 8)], single_domain_dialogue_data[int(num * 8):int(num * 9)], single_domain_dialogue_data[int(num * 9):]
        elif self.style == CRSLab_STYLE:
            num = len(single_domain_dialogue_data) / 10
            json.dump(single_domain_dialogue_data, open("ali_crslab_result/" + folder_name + "/full_data.json", "w", encoding="UTF-8"), ensure_ascii=False)
            json.dump(single_domain_dialogue_data[:int(num * 8)], open("ali_crslab_result/" + folder_name + "/train_data.json", "w", encoding="UTF-8"), ensure_ascii=False)
            json.dump(single_domain_dialogue_data[int(num * 8):int(num * 9)], open("ali_crslab_result/" + folder_name + "/valid_data.json", "w", encoding="UTF-8"), ensure_ascii=False)
            json.dump(single_domain_dialogue_data[int(num * 9):], open("ali_crslab_result/" + folder_name + "/test_data.json", "w", encoding="UTF-8"), ensure_ascii=False)

            return single_domain_dialogue_data[:int(num * 8)], single_domain_dialogue_data[int(num * 8):int(num * 9)], single_domain_dialogue_data[int(num * 9):]
        elif self.style == GPT2_STYLE:
            num = len(single_domain_dialogue_data) / 10
            json.dump(single_domain_dialogue_data, open("ali_gpt2_result/" + folder_name + "/full_data.json", "w", encoding="UTF-8"), ensure_ascii=False)
            json.dump(single_domain_dialogue_data[:int(num * 8)], open("ali_gpt2_result/" + folder_name + "/train_data.json", "w", encoding="UTF-8"), ensure_ascii=False)
            json.dump(single_domain_dialogue_data[int(num * 8):int(num * 9)], open("ali_gpt2_result/" + folder_name + "/valid_data.json", "w", encoding="UTF-8"), ensure_ascii=False)
            json.dump(single_domain_dialogue_data[int(num * 9):], open("ali_gpt2_result/" + folder_name + "/test_data.json", "w", encoding="UTF-8"), ensure_ascii=False)

            return single_domain_dialogue_data[:int(num * 8)], single_domain_dialogue_data[int(num * 8):int(num * 9)], single_domain_dialogue_data[int(num * 9):]

    def dump_data_ntrd(self):
        t_list = []
        v_list = []
        a_list = []
        for domain in domain_list:
            t, v, a = self.dump_dialogue_data(domain, self.dialogue_data[domain])
            t_list.extend(t)
            v_list.extend(v)
            a_list.extend(a)
        self.dump_dialogue_data("全部行业", t_list + v_list + a_list)

        pickle.dump(self.kg_item, open("ali_ntrd_result/subkg.pkl", "wb"))

        pickle.dump(self.entity_name_index_dict, open("ali_ntrd_result/entity2entityId.pkl", "wb"))

        pickle.dump({item_id: item_id for item_id in self.item_id_dict.keys()}, open("ali_ntrd_result/id2entity.pkl", "wb"))

        pickle.dump({item_id: index for item_id, index in self.item_id_dict.items()}, open("ali_ntrd_result/movieID2selection_label.pkl", "wb"))

        pickle.dump(self.text_dict, open("ali_ntrd_result/text_dict.pkl", "wb"))

        pickle.dump([self.entity_name2index(item_id) for item_id in self.item_id_dict.keys()], open("ali_ntrd_result/movie_ids.pkl", "wb"))

    def dump_data_crslab(self):
        t_list = []
        v_list = []
        a_list = []
        for domain in domain_list:
            t, v, a = self.dump_dialogue_data(domain, self.dialogue_data[domain])
            t_list.extend(t)
            v_list.extend(v)
            a_list.extend(a)
        self.dump_dialogue_data("全部行业", t_list + v_list + a_list)

        json.dump(self.kg_item, open("ali_crslab_result/subkg.json", "w", encoding="UTF-8"), ensure_ascii=False)
        json.dump(self.entity_name_index_dict, open("ali_crslab_result/entity2id.json", "w", encoding="UTF-8"), ensure_ascii=False)

        json.dump([self.entity_name2index(item_id) for item_id in self.item_id_dict.keys()], open("ali_crslab_result/movie_ids.json", "w", encoding="UTF-8"), ensure_ascii=False)

        token2id = json.load(open("ltp_vocab_list/带special_token的单词表.json"))
        json.dump(token2id, open("ali_crslab_result/token2id.json", "w", encoding="UTF-8"), ensure_ascii=False)


    def dump_data_gpt2(self):
        t_list = []
        v_list = []
        a_list = []
        for domain in domain_list:
            t, v, a = self.dump_dialogue_data(domain, self.dialogue_data[domain])
            t_list.extend(t)
            v_list.extend(v)
            a_list.extend(a)
        self.dump_dialogue_data("全部行业", t_list + v_list + a_list)

        json.dump(self.kg_item, open("ali_gpt2_result/subkg.json", "w", encoding="UTF-8"), ensure_ascii=False)
        json.dump(self.entity_name_index_dict, open("ali_gpt2_result/entity2id.json", "w", encoding="UTF-8"), ensure_ascii=False)

        json.dump([self.entity_name2index(item_id) for item_id in self.item_id_dict.keys()], open("ali_gpt2_result/movie_ids.json", "w", encoding="UTF-8"), ensure_ascii=False)

        vocab_file = open("./gpt2/vocab.txt", "r")
        vocab_list = vocab_file.readlines()
        token2id = {vocab_list[i][:-1]: i for i in range(len(vocab_list))}
        json.dump(token2id, open("ali_gpt2_result/token2id.json", "w", encoding="UTF-8"), ensure_ascii=False)


if __name__ == "__main__":
    style = NTRD_STYLE
    # style = CRSLab_STYLE
    # style = GPT2_STYLE
    
    DataProcess(style)
