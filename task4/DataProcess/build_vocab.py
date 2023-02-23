import json
import pickle

from gensim import corpora
from ltp import LTP
from tqdm import tqdm
from transformers import AutoTokenizer

# tokenize_tool = "ltp"
tokenize_tool = "gpt2"

dialogue_data = json.load(open("std_ali_data/air_dataset_0210_encrypt.json", "r"))

sentences = []
item_ids = set()

for session in dialogue_data:
    for utterance in session["dialogue"]:
        if utterance["rec_item_id"].__len__() != 0:
            temp_str = ""
            for item_id in utterance["rec_item_id"]:
                item_ids.add(item_id)
                temp_str += "@" + item_id + " "

            if utterance["send_content"] == "仅发送商品链接":
                sentences.append(temp_str[:-1])
            else:
                sentences.append(temp_str + utterance["send_content"])
        else:
            sentences.append(utterance["send_content"])

corpus = []
if tokenize_tool == "ltp":
    ltp = LTP("LTP/legacy")
    ltp.add_words(["@" + item_id for item_id in item_ids])
    corpus = ltp.pipeline(sentences, tasks=["cws"])[0]
elif tokenize_tool == "gpt2":
    tokenizer = AutoTokenizer.from_pretrained("gpt2/")
    for sen in tqdm(sentences):
        corpus.append(tokenizer(sen).tokens()[1:-1])
else:
    exit(-1)

f1 = open(tokenize_tool + "_vocab_list/raw_sen.txt", "w")
f1.write("\n".join(sentences))
f1.close()
f2 = open(tokenize_tool + "_vocab_list/tokenize_sen.txt", "w")
f2.write("\n".join(["|".join(i) for i in corpus]))
f2.close()

sen_tokenize_sen_dict = {}
for i in range(len(sentences)):
    sen_tokenize_sen_dict[sentences[i]] = corpus[i]
pickle.dump(sen_tokenize_sen_dict, open(tokenize_tool + "_vocab_list/sen_tokenize_sen_dict.pkl", "wb"))

if tokenize_tool == "ltp":
    vocab = corpora.Dictionary(corpus).token2id

    vocab_with_special_token = {"__pad__": 0, "__start__": 1, "__end__": 2, "__unk__": 3}

    for token, index in vocab.items():
        vocab_with_special_token[token] = index + 4
    vocab_with_special_token["_split_"] = len(vocab_with_special_token)

    json.dump(vocab, open("ltp_vocab_list/单词表.json", "w", encoding="UTF-8"), ensure_ascii=False)
    json.dump(vocab_with_special_token, open("ltp_vocab_list/带special_token的单词表.json", "w", encoding="UTF-8"), ensure_ascii=False)
