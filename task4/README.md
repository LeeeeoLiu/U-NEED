# 任务简介 Task Introduction
从总任务的本质属性来看，对话式推荐本质是推荐，推荐便需要向用户介绍信息，这也是其区别于任务型对话的关键点；从总任务的功能结构划分来看，任务4旨在基于给定的信息生成高质量的回复。因此我们在选择基线方法时考虑了两种常见将给定信息加入到解码过程的方法：隐变量方法和特定解码机制。

From the perspective of the essential attributes of the total task, the essence of conversational recommendation is recommendation, and recommendation needs to introduce information to users, which is also the key point different from task-oriented dialogue; from the perspective of the functional structure of the total task, task 4 aims to generate high-quality responses based on the given information. We therefore considered two common ways to incorporate given information into the decoding process when choosing our baseline method: latent variable methods and specific decoding mechanisms.

具体而言，基线方法如下：
- transformer 将商品ID也视为token进行预测学习；
- GPT-2 使用预训练模型学习对话；
- KBRD 提出了推荐感知的回复生成模块，将用户表示映射到词表权重，使模型在解码词时更容易解码出推荐电影相关词的概率。
- NTRD 首先生成带有特殊token的回复模板，然后将推荐的结果替换特殊token。

Specifically, the baseline methods are as follows:
- transformer treat item ID as a token to predict;
- GPT-2 uses pre-trained language model;
- KBRD proposes a recommendation-aware reply generation module, which maps user representations to vocabulary weights, making it easier for the model to decode the probability of recommended movie-related words when decoding words.
- NTRD first generates a reply template with a special token, and then replaces the special token with the recommended result.

# 评价指标 Evaluation Metric
1. 自动化评价指标：Dist-{1,2,3,4}，Bleu-{1,2,3,4}；
2. 人工评价指标：
- 信息量：生成回复中包含商品信息的量；
- 相关性：生成回复中的商品信息与客服回复中的商品信息的相关程度；

1. Automatic evaluation metric: Dist-{1,2,3,4}, Bleu-{1,2,3,4};
2. Manual evaluation metric:
- Information amount: the amount of item information contained in the generated response;
- Correlation: the degree of correlation between the item information in the generated response and the item information in the customer service staff response;

# 代码运行 Run Code
## 下载预训练模型 Download Pretrain language model
下载预训练模型[GPT2](https://huggingface.co/uer/gpt2-chinese-cluecorpussmall/tree/main)至DataProcess/gpt2、CRSLab\data\model\pretrain\gpt2\，注意保留CRSLab\data\model\pretrain\gpt2\.built文件

Download pretrain language model [GPT2](https://huggingface.co/uer/gpt2-chinese-cluecorpussmall/tree/main) to DataProcess/gpt2, CRSLab\data\model\pretrain\gpt2\, be careful not to cover CRSLab\data\ model\pretrain\gpt2\.built file

## 数据预处理 DataProcess
在DataProcess/文件夹下，首先修改并运行build_vocab.py文件：
- 设置tokenize_tool = "gpt2"，对原始对话文本进行分词，结果会保存至DataProcess\gpt2_vocab_list；
- 设置tokenize_tool = "ltp"，对原始对话文本进行分词并得到单词表，结果会保存至DataProcess\ltp_vocab_list；

Under DataProcess/, modify and run build_vocab.py file:
- Set tokenize_tool = "gpt2" to segment the original dialogue text, and the result will be saved to DataProcess\gpt2_vocab_list;
- Set tokenize_tool = "ltp" to segment the original dialogue text and get the vocabulary list, the result will be saved to DataProcess\ltp_vocab_list;

然后修改并运行main.py文件：
- 设置style = NTRD_STYLE，获得用于NTRD模型的训练数据，结果保存至DataProcess\ali_ntrd_result中；
- 设置style = CRSLab_STYLE，获得用于KBRD、Transformer模型的训练数据，结果保存至DataProcess\ali_crslab_result中；
- 设置style = GPT2_STYLE，获得用于GPT-2模型的训练数据，结果保存至DataProcess\ali_gpt2_result中；

Then modify and run the main.py file:
- Set style = NTRD_STYLE to obtain training data for the NTRD model, and the result will be saved to DataProcess\ali_ntrd_result;
- Set style = CRSLab_STYLE to obtain training data for KBRD and Transformer models,  and the result will be saved to DataProcess\ali_crslab_result;
- Set style = GPT2_STYLE to obtain training data for the GPT-2 model, and the result will be saved to DataProcess\ali_gpt2_result;

## NTRD
该模型代码来自于[NTRD](https://github.com/jokieleung/NTRD)。

The code of NTRD model is borrowed from [NTRD](https://github.com/jokieleung/NTRD).

将处理好的数据复制至./NTRD/data/文件夹下。

Copy the processed data to the ./NTRD/data/ folder.

```bash
cp ../DataProcess/ali_ntrd_result/全部行业/* ../DataProcess/ali_ntrd_result/* ./data/
```

开始训练 

Begin to train.

```bash
python run.py --learningrate 0.0001 --max_c_length 1024 --max_r_length 100 --beam 3 --momentum 0.9 --dropout 0.2 --attention_dropout 0.2 --relu_dropout 0.2  --n_heads 10 --batch_size 32 --n_layers 4 --ffn_size 300 --embedding_size 300 --epoch 300 --gradient_clip 0.5
```

生成结果保存在cases文件夹中。

The generate results will be saved to cases/.

## CRSLab
该模型代码来自于[CRSLab](https://github.com/RUCAIBox/CRSLab)。

This code is borrowed from [CRSLab](https://github.com/RUCAIBox/CRSLab).

将处理好的数据复制至./CRSLab/data/dataset/uneod/文件夹下。

Copy the processed data to the ./CRSLab/data/dataset/uneod/ folder.

```bash
cp ../DataProcess/ali_crslab_result/全部行业/* ../DataProcess/ali_crslab_result/* ./data/dataset/uneod/ltp

cp ../DataProcess/ali_gpt2_result/全部行业/* ../DataProcess/ali_gpt2_result/* ./data/dataset/uneod/gpt2
```

训练Transformer模型

Train Transformer

```bash
python run_crslab.py -c ./config/conversation/transformer/uneod.yaml --save_system --gpu 0
```

训练KBRD模型

Train KBRD

```bash
python run_crslab.py -c ./config/crs/kbrd/uneod.yaml --save_system --gpu 0
```

训练GPT-2模型

Train GPT-2

```bash
python run_crslab.py -c ./config/conversation/gpt2/uneod.yaml --save_system --gpu 0
```

生成结果均保存在gen_result中。

The generate results will be saved to gen_result/.