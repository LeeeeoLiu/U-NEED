# 任务简介 Task Introduction
提问任务旨在从当前的对话历史中挖掘用户的潜在可能需求，来选择若干个能够引导出更多用户需求信息的属性。

Elicitation task aims to mine the potential needs of users from current dialogue history to select several attributes that can lead to more information about user needs.

具体而言，基线方法如下：
1. 建模为多标签分类任务，通过Bert对文本进行编码，然后直接预测每个属性的概率，若概率大于0.5，则选为本轮需要提问的属性；
2. 建模为序列生成任务，利用自回归生成来建模属性集合之间的依赖关系

Specifically, two baseline methods are as follows:
1. Model the task as a multi-label classification task. Encode the text with Bert, and then directly predict the probability of each attribute. If the probability is greater than 0.5, it will be selected as the attribute that needs to be asked in this turn;
2. Model the task as a sequence generation task, using autoregressive generation to model dependencies between attribute sets.

# 评价指标 Evaluation Metric
Precison、Recall、F1

# 数据构造 Data Preparation
在一个对话中，当遇到标签为"系统提问"的utterance，则构造一条数据。标签为该utterance包含的所有询问的属性，输入为当前轮之前的对话历史的拼接+[SEP]+对话历史提到的所有属性。

示例如下：


```json
{
    "domain": "鞋类行业",
    "sid": "1b656eb8b3759ec61f73a4be3ebd8ceb",
    "sellerid": "520557925",
    "userid": "2207727033661",
    "context": [
      "有运动鞋推荐吗",
      "您平时穿多大尺码呢，运动量多少公里呢",
      "学生",
      "每天都要跑步",
      "38"
    ],
    "attr_path": [
      {
        "品类": "运动鞋"
      },
      {
        "鞋码": "",
        "跑步强度": ""
      },
      {
        "人群": "学生"
      },
      {
        "使用场景": "跑步"
      },
      {
        "鞋码": "38"
      }
    ],
    "text": "有运动鞋推荐吗 您平时穿多大尺码呢，运动量多少公里呢 学生 每天都要跑步 38[SEP]83 67 75 35 16 67",
    "label": "0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0",
    "attr_seq": [1,65,69,2,0,0,0],
    "raw_label": {
      "颜色": "",
      "价位": ""
    },
    "reply": "您对颜色和价位有没又需求呢"
  }
```

本任务的输入即为"text"字段，标签有两种，第一种是label，表示为one-hot向量，在需要询问的属性位置数值为1，其余为0；第二种是attr_seq，表示为属性index的序列，其中1表示SOS生成的起始符号，2表示EOS生成的终止符，0表示PAD。这两种标签分别对应上面提到的两种基线方法。

# 代码运行
## 数据处理
指定data_process.py中的`train_path`,`dev_path`,`test_path`，然后运行data_process.py。程序将会自动从raw_data中抽取任务2相关的数据，并构造成符合模型输入，输出的格式，同时也会形成各领域的train,dev,test数据，打印数据统计情况。

## baseline
两个baseline的超参数配置都在config.yaml中
### 多标签分类
代码文件为multi_label.py，数据构造后之后，直接运行即可训练模型。命令行参数有：
- -dm（领域，其中all为全领域
- -test（直接测试
- -ckpt（指定测试的checkpoints，若不指定则使用train得到的最优模型，即output_dir）
- -suffix（改变模型和日志保存路径，将会保存到{output_dir}/{suffix}/{domain}）

### 序列生成
代码文件为attr_generate.py，数据构造完之后，直接运行即可训练模型。命令行参数与multi_label.py基本一致

注：两个baseline都需要先在全领域all上训练，然后在各领域训练将会导入从all领域上训练得到的最优模型