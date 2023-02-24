# 标注目标 Annotation Target
从店小蜜的人人对话中，抽取导购相关的轮次，并对每个轮次的内容进行结构化理解和抽取，最终构建一份交互导购标准数据集。

From DianXiaoMi's Human-Human conversations, we extract the turns related to shopping, and the content of each turn is structured and extracted, and finally construct a standard dataset of interactive pre-sale diaglogues.

# 生产标准 Annotation Standard
整体步骤如下：给定一通对话，从中抽取出【核心导购轮次】，并针对核心导购轮次进行多个步骤的标注，包括对话行为识别、属性识别、推荐商品id、对话内容修正。

The overall steps are as follows: Given a conversation, extract the [core pre-sale turns] from it, and mark the core pre-sale turn in multiple steps, including dialogue behavior identification, attribute identification, recommended item id, and dialogue content correction.

## 步骤一【是否导购轮次】 Step 1 Whether pre-sale turns
核心导购轮次定义：
- 以用户表达需求开始，用户与系统进行多轮关于商品属性的问答，其中系统会产生推荐行为，用户可能根据推荐结果给出反馈，直到推荐行为结束。注意，普通问候、商品类问答、物流售后等均不算在内。

Definition of core pre-sale turns:
- Start with the user expressing need, the user and the system conduct multiple turns of questions and answers about item attributes, in which the system will generate a recommendation behavior, and the user may give feedback based on the recommendation result until the end of the recommendation behavior. Note that general greetings, item-related questions and answers, logistics after-sales, etc. are not included.

场景核心导购轮次构成的模式：
- 用户需求-系统提问-用户回答-系统推荐/系统解释-用户反馈-系统推荐/系统解释
- 用户需求-系统提问-用户回答-系统推荐/系统解释-用户反馈-系统提问-用户回答-系统推荐/系统解释

The pattern of core pre-sale turns in the scenario:
- User needs-system question-user answer-system recommendation/system explanation-user feedback-system recommendation/system explanation
- User needs-system question-user answer-system recommendation/system explanation-user feedback-system question-user answer-system recommendation/system explanation

标注注意事项：
- 如果整个对话中没有符合如上模式，例如仅包含用户需求-系统推荐，则整个对话都打-1，不进行后面标注步骤。
- 对于一个对话中有符合要求的部分导购轮次，挑出来，属于核心轮次打1，不是则留空

Annoatation Notes:
- If the above pattern is not met in the entire dialogue, for example, only user needs-system recommendations are included, then the entire dialogue will be marked with -1, and the subsequent labeling steps will not be performed.
- For some pre-sale turns that meet the requirements in a dialogue, pick them out, if they belong to the core turns, hit 1, if not, leave blank

## 步骤二【对话行为】 Step 2 Dialogue behavior
注意：该任务仅针对步骤一为1的轮次，需要进行标注。

Note: This task is only for turns where step 1 is 1 and needs to be marked.

对话行为定义：

Dialogue Behavior Definition:

- 用户需求
   - 用户最初表达自己的购买需求。一般出现在核心导购轮次最靠前的位置
   - 例如：我想要一款补水的面霜，推荐一下；50岁左右用哪个眼霜好
   - 注意：
      - 如果用户在多个轮次表达了相同的诉求（可能因为不同客服接待导致重复表达），选择最近的轮次进行标注。

- User needs
    - Users initially express their purchase needs. It usually appears at the front of the core pre-sale turns.
    - For example: I want a hydrating face cream, please recommend; which eye cream is best for around 50 years old
    - Notice:
       - If the user expresses the same appeal in multiple turns (it may be repeated due to different customer service reception), select the most recent turns to mark.

- 系统提问
   - 客服询问某个特定属性，意在获取用户的进一步偏好。一般出现在用户需求后
   - 例如：请问您肤质是什么呢；方便透露下您的年龄吗
   - 注意：
      - 客服的提问必须关联到第三个sheet【1美妆_属性】中提到的内容，如果只是普通的询问，如有什么可以帮您呢、或者问不在范围内的属性，则不标注出来。

- System question
    - The customer service staff asks for a specific attribute in order to obtain the user's further preferences. Generally appear after user needs
    - For example: what is your skin type; can you tell me your age?
    - Notice:
       - Questions from customer service staff must be related to the content mentioned in the third sheet. If it is just a general inquiry, if there is anything I can help you with, or ask for attributes that are not within the scope, then do not mark it .

- 用户回答
   - 用户回答客服询问的某个属性对应的具体偏好。一般出现在系统提问后面的用户发言
   - 例如：我皮肤有点油；今年25岁

- User answer
    - The user's specific preference corresponding to a certain attribute of the customer service inquiry. The user answer that usually appears after the system question
    - Example: I have oily skin; I am 25 years old

- 系统推荐
   - 客服推荐某款商品，必须要包含某商品id，可能部分包含营销话术。一般出现在用户回答或用户反馈后
   - 例如：亲，推荐您用这款商品哦id=xxxx；
   - 注意：
      - 标注时除了语义关系判断以外，还要注意时间连续性。例如第17轮虽然推荐了某个商品，但是可以看到距离用户上次发言已经过去了17分钟，这个是系统自动推荐某个活动链接，并不是客服针对用户诉求推荐的某款商品，注意甄别。

- System recommendation
    - When customer service staff recommends a certain product, it must contain a certain item id, which may partially contain marketing words. Usually appear after user answers or user feedback.
    - For example: Dear, I recommend you to use this item id=xxxx;
    - Notice:
       - In addition to judging the semantic relationship when labeling, attention should also be paid to the continuity of time. For example, although a product was recommended in the 17th turn, it can be seen that 17 minutes have passed since the user’s last utterance. This is an activity link automatically recommended by the system, not a item recommended by customer service staff in response to the user’s appeal. Pay attention to screening.

- 用户反馈
   - 用户针对上文的客服推荐，反馈了进一步的求购诉求。一般出现在系统更推荐后
   - 例如：这款会不会有点油腻了，我要干爽点的；其他适合的系列有推荐么
   - 注意：
      - 用户反馈和用户回答的区别是，一般在某个推荐过后，用户的诉求都认为是用户反馈。在第一次推荐之前，用户针对客服的提问所做的回答，一般都是用户回答

- customer feedback
    - The user gave feedback on further purchase demands based on the customer service staff recommendation above. Generally appear after the system recommends
    - For example: Will this one be a bit greasy, I want something dry; do you have any recommendations for other suitable series?
    - Notice:
       - The difference between user feedback and user answer is that generally after a certain recommendation, the user's appeal is considered user feedback. Before the first recommendation, the user's answer to the customer service staff's question is generally the user's answer

- 系统解释
   - 客服在推荐商品过程中的仅文字介绍话术。一般会出现在系统推荐的前后
   - 例如：这款加入了精华成分，是专门针对您这边的皱纹的情况的。
   - 注意：
      - 普通的话术，如您可以试试哦，这种不包含商品类的介绍不需要纳入。一般系统解释指的是推荐这款商品的理由，比如包含了针对用户某个需求的商品特色信息介绍。

- System explanation
    - The text-only introduction from customer service staff in the process of recommending item. Generally, it will appear before and after the system recommendation.
    - For example: this product contains essence ingredients, which are specially aimed at the wrinkles on your side.
    - Notice:
       - Ordinary words, like you can try it, this kind of introduction that does not include commodities does not need to be included. The general system explanation refers to the reason for recommending this item, for example, it includes the introduction of item feature information for a user's needs.

标注注意事项：

- 对于核心导购轮次为1的，必须标注对话行为，如果不是则不用选择
- 对话行为必须在以上六个中选择一个，在下拉框内进行选择

Annotation Notice:

- For the core pre-sale turns of 1, the dialogue behavior must be marked, if not, do not choose
- Dialogue behavior must choose one of the above six, choose in the drop-down box

## 步骤三【属性识别】Step 3 Attribute Identification
> 提供了一个算法识别的属性结果，但很大概率有遗漏的情况，仅供参考

> Provides an attribute result identified by an algorithm, but there is a high probability that there will be omissions, which are for reference only.

注意：该任务仅针对任务二为用户需求、系统提问、用户回答、用户反馈的轮次，需要进行标注。

Note: This step only needs to be marked for the turns in which step 2 is user requirements, system questions, user answers, and user feedback.

属性定义：
- 需要识别句子中提到的属性名和抽取到的属性值。建议先提前过一遍例子，了解需要抽取哪些内容，这些属性分别代表什么含义。
- 正常的标注格式是属性名：属性值，例如：【肤质：干性】
- 如果一个属性有多个值，用逗号分隔，例如：【功效：保湿，祛痘印】
- 如果一个轮次问了多个属性，则用##分隔，例如：【肤质：干性##功效：保湿，祛痘印】
- 如果只询问某个属性名，但是没有提供候选值，则冒号后留空，例如【肤质：】

Attribute definition:
- It is necessary to identify the attribute names and mentioned in the sentence and the extracted attribute values. It is recommended to go through examples in advance to understand what content needs to be extracted and what these attributes represent.
- The normal annotation format is attribute name: attribute value, for example: [skin type: dry]
- If an attribute has multiple values, separate them with commas, for example: [Efficacy: Moisturizing, Acne Removal]
- If multiple attributes are asked in one turn, use ## to separate them, for example: [Skin Type: Dry ##Efficacy: Moisturizing, Acne Removal]
- If you only ask for a certain attribute name, but no candidate value is provided, leave it blank after the colon, for example [skin quality:]

标注注意事项：
- 针对用户说的否定内容，需要把否定词也加入，如【肌肤问题：不是过敏】
- 抽取属性值的时候，相邻表达的尽量抽取完整，不要断开或者遗漏，例如这里算法抽取的结果遗漏了屏障，而且把泛红干痒给分开了，应该是完整的【肌肤问题：屏障受损，泛红干痒】

Annotation Notes:
- For the negative content that the user said, it is necessary to add negative words, such as [skin problem: not allergies]
- When extracting attribute values, try to extract the adjacent expressions as completely as possible, and do not disconnect or omit them. For example, the result of the algorithm extraction here misses the barrier, and separates the redness, dryness and itching. It should be complete [Skin Problem: Barrier Damaged, red, dry and itchy]

## 步骤四【推荐商品id】Step 4 Recommend Item Id
> 提供了一个规则抽取的商品id在K列，不会出错，但可能会存在遗漏情况，仅供参考

> The item id extracted by a algorithm is provided, there will be no error, but there may be omissions, it is for reference only.

注意：该步骤仅针对步骤二为系统推荐的轮次，需要进行标注。

Note: This step is only for the turn which is labeled as system recommend in step 2 and needs to be marked.

商品id定义：
- 将句子中id=xxxxx中xxxx这串数字抽取出来，写在推荐商品id处
- 注意，如果一句句子中包含多个商品id，用逗号分隔开

Item id definition:
- Extract the string of numbers xxxx from id=xxxxx in the sentence and write it in the recommended item id
- Note that if a sentence contains multiple item ids, separate them with commas

## 步骤五【对话内容修正】Step 5 Dialogue Content Correction
注意：该步骤仅针对步骤二为系统推荐和系统解释的轮次，需要进行标注。

Note: This step is only for the turns of system recommendation and system interpretation in step 2, and needs to be marked.

对话内容修正目的：
- 将敏感的某个商品名称、某个商品id进行脱敏，用【商品名称】、【商品链接】替换
- 将无关的店铺活动、增品信息等删除，仅保留商品相关的推荐理由（即系统解释）
- 例如：
   - 原句子：清香400ml+清香100ml【活动时间】10月8日20:00:00-10月19日23:59:59【券后价】66.9元【功效】水润深护，改善肌肤干燥，更适用于皮肤较干燥，有脱皮问题的宝宝哦~----------------------------ps：活动页面不断更新，具体以拍下页面为准~【商品链接】
   - 修改后的句子：【功效】水润深护，改善肌肤干燥，更适用于皮肤较干燥，有脱皮问题的宝宝哦【商品链接】

Dialogue Content Correction Purpose:
- Desensitize a sensitive item name and item id, and replace it with [product name] and [product link]
- Delete irrelevant store activities, item addition information, etc., and only keep item-related recommendation reasons (that is, system explanations)
- For example:
    - Original sentence: Fragrance 400ml+Fragrance 100ml [Activity time] October 8th 20:00:00-October 19th 23:59:59 [Price after coupon] 66.9 yuan [Efficacy] Moisturizing and deep care, improve skin Dry, more suitable for babies with dry skin and peeling problems~------------------------------- ps: The event page is constantly updated , please refer to the photographed page for details~[item link]
    - Modified sentence: [Efficacy] Moisturizing and deep care, improving dry skin, more suitable for babies with dry skin and peeling problems [Product Link]

# 验收标准 Acceptance Criteria
抽检结果错误率低于5%，验收通过。

The error rate of random inspection results is less than 5%, and the acceptance is passed.