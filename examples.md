U-NEED为中文数据集，以下是我们为每个行业提供的样例。

The language of U-NEED is Chinese, and we provide an example for each category as follows.

# 美妆 Beauty
```json
{
  "domain": "美妆行业",
  "sid": "03c621cb07fb9dc745e35fecd79e1d56",
  "sellerid": "2830038155",
  "userid": "2201523477005",
  "ds": "20180930",
  "dialogue": [
    {
      "seq_no": 1,
      "time": "2018-09-30 15:57:00",
      "sender_type": "用户",
      "send_content": "你们店有没有套装",
      "act_tag": "用户需求",
      "attributes": [
        {
          "key": "品类",
          "value": "套装",
          "start": 6,
          "exclusive_end": 8
        }
      ],
      "rec_item_id": [],
      "rec_item_info": []
    },
    {
      "seq_no": 2,
      "time": "2018-09-30 15:57:14",
      "sender_type": "客服",
      "send_content": "您想要什么类型的呢",
      "act_tag": "系统提问",
      "attributes": [
        {
          "key": "功效",
          "value": "",
          "start": -1,
          "exclusive_end": -1
        }
      ],
      "rec_item_id": [],
      "rec_item_info": []
    },
    {
      "seq_no": 3,
      "time": "2018-09-30 15:57:34",
      "sender_type": "用户",
      "send_content": "补水",
      "act_tag": "用户回答",
      "attributes": [
        {
          "key": "功效",
          "value": "补水",
          "start": 0,
          "exclusive_end": 2
        }
      ],
      "rec_item_id": [],
      "rec_item_info": []
    },
    {
      "seq_no": 4,
      "time": "2018-09-30 15:57:47",
      "sender_type": "客服",
      "send_content": "水乳吗亲亲",
      "act_tag": "系统提问",
      "attributes": [
        {
          "key": "品类",
          "value": "水乳",
          "start": 0,
          "exclusive_end": 2
        }
      ],
      "rec_item_id": [],
      "rec_item_info": []
    },
    {
      "seq_no": 5,
      "time": "2018-09-30 15:57:50",
      "sender_type": "用户",
      "send_content": "嗯，需要水乳",
      "act_tag": "用户回答",
      "attributes": [
        {
          "key": "品类",
          "value": "水乳",
          "start": 4,
          "exclusive_end": 6
        }
      ],
      "rec_item_id": [],
      "rec_item_info": []
    },
    {
      "seq_no": 6,
      "time": "2018-09-30 15:58:00",
      "sender_type": "客服",
      "send_content": "亲亲需要祛痘的吗",
      "act_tag": "系统提问",
      "attributes": [
      {
      "key": "功效",
      "value": "祛痘",
      "start": 4,
      "exclusive_end": 6
    }
      ],
      "rec_item_id": [],
      "rec_item_info": []
    },
      {
      "seq_no": 7,
      "time": "2018-09-30 15:58:09",
      "sender_type": "用户",
      "send_content": "需要祛痘的",
      "act_tag": "用户回答",
      "attributes": [
      {
      "key": "功效",
      "value": "祛痘",
      "start": 2,
      "exclusive_end": 4
    }
      ],
      "rec_item_id": [],
      "rec_item_info": []
    },
      {
      "seq_no": 8,
      "time": "2018-09-30 15:58:26",
      "sender_type": "客服",
      "send_content": "亲亲是想解决红肿痘痘还是闭口的么",
      "act_tag": "系统提问",
      "attributes": [
      {
      "key": "肌肤问题",
      "value": "红肿痘痘",
      "start": 6,
      "exclusive_end": 10
    },
      {
      "key": "肌肤问题",
      "value": "闭口",
      "start": 12,
      "exclusive_end": 14
    }
      ],
      "rec_item_id": [],
      "rec_item_info": []
    },
      {
      "seq_no": 9,
      "time": "2018-09-30 15:58:44",
      "sender_type": "用户",
      "send_content": "痘印",
      "act_tag": "用户回答",
      "attributes": [
      {
      "key": "肌肤问题",
      "value": "痘印",
      "start": 0,
      "exclusive_end": 2
    }
      ],
      "rec_item_id": [],
      "rec_item_info": []
    },
      {
      "seq_no": 10,
      "time": "2018-09-30 16:00:07",
      "sender_type": "客服",
      "send_content": "亲亲是敏感肌肤吗",
      "act_tag": "系统提问",
      "attributes": [
      {
      "key": "肤质",
      "value": "敏感肌肤",
      "start": 3,
      "exclusive_end": 7
    }
      ],
      "rec_item_id": [],
      "rec_item_info": []
    },
      {
      "seq_no": 11,
      "time": "2018-09-30 16:01:02",
      "sender_type": "用户",
      "send_content": "有点敏感",
      "act_tag": "用户回答",
      "attributes": [
      {
      "key": "肌肤问题",
      "value": "有点敏感",
      "start": 0,
      "exclusive_end": 4
    }
      ],
      "rec_item_id": [],
      "rec_item_info": []
    },
      {
      "seq_no": 12,
      "time": "2018-09-30 16:01:16",
      "sender_type": "客服",
      "send_content": "仅发送商品链接",
      "act_tag": "系统推荐",
      "attributes": [],
      "rec_item_id": [
      "565208902623"
      ],
      "rec_item_info": [
      "565208902623@1801"
      ]
    },
      {
      "seq_no": 13,
      "time": "2018-09-30 16:01:25",
      "sender_type": "客服",
      "send_content": "仅发送商品链接",
      "act_tag": "系统推荐",
      "attributes": [],
      "rec_item_id": [
      "537140827307"
      ],
      "rec_item_info": [
      "537140827307@1801"
      ]
    },
      {
      "seq_no": 14,
      "time": "2018-09-30 16:01:30",
      "sender_type": "客服",
      "send_content": "这款商品富含5%维生素原B5，以及积雪草苷，可促进纤维细胞更新和淡化印痕，对于淡化6个月内的新生痘印效果很不错的哦",
      "act_tag": "系统解释",
      "attributes": [],
      "rec_item_id": [],
      "rec_item_info": []
    }
      ]
    }
```

# 服装 Fashion
```json
 {
        "domain": "服装行业",
        "sid": "46969fdb23caf3d094e690e6afc77c35",
        "sellerid": "917228782",
        "userid": "1849468581",
        "ds": "20181011",
        "dialogue": [
            {
                "seq_no": 1,
                "time": "2018-10-11 11:35:07.999996",
                "sender_type": "用户",
                "send_content": "还有没有其他款推荐",
                "act_tag": "用户需求",
                "attributes": [
                    {
                        "key": "款式",
                        "value": "",
                        "start": -1,
                        "exclusive_end": -1
                    }
                ],
                "rec_item_id": [],
                "rec_item_info": []
            },
            {
                "seq_no": 2,
                "time": "2018-10-11 11:35:34.000002",
                "sender_type": "客服",
                "send_content": "要什么材质的呢亲爱哒",
                "act_tag": "系统提问",
                "attributes": [
                    {
                        "key": "材质",
                        "value": "",
                        "start": -1,
                        "exclusive_end": -1
                    }
                ],
                "rec_item_id": [],
                "rec_item_info": []
            },
            {
                "seq_no": 3,
                "time": "2018-10-11 11:36:08.000001",
                "sender_type": "用户",
                "send_content": "就是冰丝的那种",
                "act_tag": "用户回答",
                "attributes": [
                    {
                        "key": "材质",
                        "value": "冰丝",
                        "start": 2,
                        "exclusive_end": 4
                    }
                ],
                "rec_item_id": [],
                "rec_item_info": []
            },
            {
                "seq_no": 4,
                "time": "2018-10-11 11:36:54",
                "sender_type": "客服",
                "send_content": "仅发送商品链接",
                "act_tag": "系统推荐",
                "attributes": [],
                "rec_item_id": [
                    "669362157284"
                ],
                "rec_item_info": [
                    "669362157284@1625"
                ]
            },
            {
                "seq_no": 5,
                "time": "2018-10-11 11:37:03.999996",
                "sender_type": "客服",
                "send_content": "这个是莫代尔材质冰丝触感的",
                "act_tag": "系统解释",
                "attributes": [],
                "rec_item_id": [],
                "rec_item_info": []
            },
            {
                "seq_no": 6,
                "time": "2018-10-11 11:37:10.000001",
                "sender_type": "客服",
                "send_content": "仅发送商品链接",
                "act_tag": "系统推荐",
                "attributes": [],
                "rec_item_id": [
                    "668410245231"
                ],
                "rec_item_info": [
                    "668410245231@1625"
                ]
            },
            {
                "seq_no": 7,
                "time": "2018-10-11 11:37:13",
                "sender_type": "客服",
                "send_content": "这个是冰丝材质的",
                "act_tag": "系统解释",
                "attributes": [],
                "rec_item_id": [],
                "rec_item_info": []
            },
            {
                "seq_no": 8,
                "time": "2018-10-11 11:40:35.999996",
                "sender_type": "用户",
                "send_content": "冬天了穿哪款好点",
                "act_tag": "用户反馈",
                "attributes": [
                    {
                        "key": "季节",
                        "value": "冬天",
                        "start": 0,
                        "exclusive_end": 2
                    }
                ],
                "rec_item_id": [],
                "rec_item_info": []
            },
            {
                "seq_no": 9,
                "time": "2018-10-11 11:41:30.999998",
                "sender_type": "用户",
                "send_content": "我儿子就喜欢穿A品牌的",
                "act_tag": "用户反馈",
                "attributes": [
                    {
                        "key": "人群",
                        "value": "儿子",
                        "start": 1,
                        "exclusive_end": 3
                    },
                    {
                        "key": "品牌",
                        "value": "A品牌",
                        "start": 7,
                        "exclusive_end": 10
                    }
                ],
                "rec_item_id": [],
                "rec_item_info": []
            },
            {
                "seq_no": 10,
                "time": "2018-10-11 11:43:17.000002",
                "sender_type": "客服",
                "send_content": "仅发送商品链接",
                "act_tag": "系统推荐",
                "attributes": [],
                "rec_item_id": [
                    "642569934258"
                ],
                "rec_item_info": [
                    "642569934258@1625"
                ]
            },
            {
                "seq_no": 11,
                "time": "2018-10-11 11:43:29.000003",
                "sender_type": "客服",
                "send_content": "冬天穿棉质的比较舒适的哦亲",
                "act_tag": "系统解释",
                "attributes": [],
                "rec_item_id": [],
                "rec_item_info": []
            },
            {
                "seq_no": 12,
                "time": "2018-10-11 11:45:37.999999",
                "sender_type": "用户",
                "send_content": "棉质的有嘛",
                "act_tag": "用户反馈",
                "attributes": [
                    {
                        "key": "材质",
                        "value": "棉质",
                        "start": 0,
                        "exclusive_end": 2
                    }
                ],
                "rec_item_id": [],
                "rec_item_info": []
            },
            {
                "seq_no": 13,
                "time": "2018-10-11 11:46:26.999996",
                "sender_type": "客服",
                "send_content": "仅发送商品链接",
                "act_tag": "系统推荐",
                "attributes": [],
                "rec_item_id": [
                    "642569934258"
                ],
                "rec_item_info": [
                    "642569934258@1625"
                ]
            },
            {
                "seq_no": 14,
                "time": "2018-10-11 11:46:30.000003",
                "sender_type": "客服",
                "send_content": "这个就是棉质的",
                "act_tag": "系统解释",
                "attributes": [],
                "rec_item_id": [],
                "rec_item_info": []
            }
        ]
    }
```

# 手机 Phones
```json
 {
        "domain": "手机行业",
        "sid": "0479375015f8f39717ebd9e1893ac341",
        "sellerid": "1114513584",
        "userid": "2911192013",
        "ds": "20181006",
        "dialogue": [
            {
                "seq_no": 1,
                "time": "2018-10-06 16:32:14.000001",
                "sender_type": "用户",
                "send_content": "老人",
                "act_tag": "用户需求",
                "attributes": [
                    {
                        "key": "人群",
                        "value": "老人",
                        "start": 0,
                        "exclusive_end": 2
                    }
                ],
                "rec_item_id": [],
                "rec_item_info": []
            },
            {
                "seq_no": 2,
                "time": "2018-10-06 16:33:36.000003",
                "sender_type": "客服",
                "send_content": "亲亲，老人平常是看视频比较多还是拍照比较多呢~",
                "act_tag": "系统提问",
                "attributes": [
                    {
                        "key": "使用场景",
                        "value": "看视频",
                        "start": 8,
                        "exclusive_end": 11
                    },
                    {
                        "key": "使用场景",
                        "value": "拍照",
                        "start": 16,
                        "exclusive_end": 18
                    }
                ],
                "rec_item_id": [],
                "rec_item_info": []
            },
            {
                "seq_no": 3,
                "time": "2018-10-06 16:35:42",
                "sender_type": "用户",
                "send_content": "买耐用的",
                "act_tag": "用户回答",
                "attributes": [
                    {
                        "key": "功能需求",
                        "value": "耐用",
                        "start": 1,
                        "exclusive_end": 3
                    }
                ],
                "rec_item_id": [],
                "rec_item_info": []
            },
            {
                "seq_no": 4,
                "time": "2018-10-06 16:36:32",
                "sender_type": "用户",
                "send_content": "买平面的吧，曲屏不经摔",
                "act_tag": "用户回答",
                "attributes": [
                    {
                        "key": "屏幕类型",
                        "value": "平面",
                        "start": 1,
                        "exclusive_end": 3
                    }
                ],
                "rec_item_id": [],
                "rec_item_info": []
            },
            {
                "seq_no": 5,
                "time": "2018-10-06 16:36:59",
                "sender_type": "客服",
                "send_content": "仅发送商品链接",
                "act_tag": "系统推荐",
                "attributes": [],
                "rec_item_id": [
                    "663552048362"
                ],
                "rec_item_info": [
                    "663552048362@1512"
                ]
            },
            {
                "seq_no": 6,
                "time": "2018-10-06 16:37:13.999999",
                "sender_type": "客服",
                "send_content": "【CPU】骁龙695处理器；/:806【屏幕尺寸】6.81英寸屏幕；/:806【机身尺寸】166.07mm（长）×75.78mm（宽）×8.05mm（厚）；/:806【屏幕分辨率】1080*2388像素；/:806【摄像头】前置1600万像素，后置4800万像素摄像头+200万像素景深摄像头+200万像素微距摄像头；/:806【电池容量】内置4800mAh（典型值）电池；/:806【充电】充电接口类型为Type-C。",
                "act_tag": "系统解释",
                "attributes": [],
                "rec_item_id": [],
                "rec_item_info": []
            },
            {
                "seq_no": 7,
                "time": "2018-10-06 16:38:06.000003",
                "sender_type": "用户",
                "send_content": "像素要好点",
                "act_tag": "用户反馈",
                "attributes": [
                    {
                        "key": "功能需求",
                        "value": "像素要好",
                        "start": 0,
                        "exclusive_end": 4
                    }
                ],
                "rec_item_id": [],
                "rec_item_info": []
            },
            {
                "seq_no": 8,
                "time": "2018-10-06 16:38:11.999999",
                "sender_type": "用户",
                "send_content": "还有电池要大",
                "act_tag": "用户反馈",
                "attributes": [
                    {
                        "key": "功能需求",
                        "value": "电池要大",
                        "start": 2,
                        "exclusive_end": 6
                    }
                ],
                "rec_item_id": [],
                "rec_item_info": []
            },
            {
                "seq_no": 9,
                "time": "2018-10-06 16:39:14",
                "sender_type": "客服",
                "send_content": "您价位的价格范围是多少呢，小二可以给您推荐适合您的款式的哦",
                "act_tag": "系统提问",
                "attributes": [
                    {
                        "key": "价位",
                        "value": "",
                        "start": -1,
                        "exclusive_end": -1
                    },
                    {
                        "key": "款式",
                        "value": "",
                        "start": -1,
                        "exclusive_end": -1
                    }
                ],
                "rec_item_id": [],
                "rec_item_info": []
            },
            {
                "seq_no": 10,
                "time": "2018-10-06 16:39:46.999996",
                "sender_type": "用户",
                "send_content": "X30和X40的价位，价格接近",
                "act_tag": "用户回答",
                "attributes": [
                    {
                        "key": "价位",
                        "value": "X30和X40的价位",
                        "start": 0,
                        "exclusive_end": 10
                    }
                ],
                "rec_item_id": [],
                "rec_item_info": []
            },
            {
                "seq_no": 11,
                "time": "2018-10-06 16:44:19",
                "sender_type": "客服",
                "send_content": "仅发送商品链接",
                "act_tag": "系统推荐",
                "attributes": [],
                "rec_item_id": [
                    "657697588314"
                ],
                "rec_item_info": [
                    "657697588314@1512"
                ]
            },
            {
                "seq_no": 12,
                "time": "2018-10-06 16:44:28.999997",
                "sender_type": "客服",
                "send_content": "您可以看看这一款呢，是直面屏手机呢~",
                "act_tag": "系统解释",
                "attributes": [],
                "rec_item_id": [],
                "rec_item_info": []
            },
            {
                "seq_no": 13,
                "time": "2018-10-06 16:44:35.000002",
                "sender_type": "客服",
                "send_content": "【CPU】天玑900处理器；/:806【屏幕尺寸】7.09英寸；/:806【机身尺寸】174.37mm（长）×84.91mm（宽）×8.3mm（厚）；/:806【屏幕分辨率】1080*2280像素；/:806【摄像头】前置800万像素，后置6400万像素+200万像素；/:806【电池容量】内置5000mAh（典型值）电池；/:806【充电】充电接口类型为Type-C。",
                "act_tag": "系统解释",
                "attributes": [],
                "rec_item_id": [],
                "rec_item_info": []
            },
            {
                "seq_no": 14,
                "time": "2018-10-06 16:45:29.000001",
                "sender_type": "客服",
                "send_content": "亲亲，这一款屏幕够大，电池是蛮不错的哈，蛮适合老人使用的呢，您可以考虑考虑呢~",
                "act_tag": "系统解释",
                "attributes": [],
                "rec_item_id": [],
                "rec_item_info": []
            }
        ]
    }
```

# 鞋类 Shoes
```json
{
        "domain": "鞋类行业",
        "sid": "0fd12e8e410e477e8ebd0ab4194e1fe1",
        "sellerid": "2129857251",
        "userid": "1887744381",
        "ds": "20180915",
        "dialogue": [
            {
                "seq_no": 1,
                "time": "2018-09-15 20:35:42.999997",
                "sender_type": "用户",
                "send_content": "给我推荐一下500左右的男鞋",
                "act_tag": "用户需求",
                "attributes": [
                    {
                        "key": "品类",
                        "value": "男鞋",
                        "start": 12,
                        "exclusive_end": 14
                    },
                    {
                        "key": "价位",
                        "value": "500左右",
                        "start": 6,
                        "exclusive_end": 11
                    }
                ],
                "rec_item_id": [],
                "rec_item_info": []
            },
            {
                "seq_no": 2,
                "time": "2018-09-15 20:37:22.000002",
                "sender_type": "客服",
                "send_content": "您需要什么款式呢？男款还是女款，尺码颜色可以大概说说客服为您更精准的推荐哦★",
                "act_tag": "系统提问",
                "attributes": [
                    {
                        "key": "人群",
                        "value": "男",
                        "start": 9,
                        "exclusive_end": 10
                    },
                    {
                        "key": "人群",
                        "value": "女",
                        "start": 13,
                        "exclusive_end": 14
                    },
                    {
                        "key": "鞋码",
                        "value": "",
                        "start": -1,
                        "exclusive_end": -1
                    },
                    {
                        "key": "颜色",
                        "value": "",
                        "start": -1,
                        "exclusive_end": -1
                    }
                ],
                "rec_item_id": [],
                "rec_item_info": []
            },
            {
                "seq_no": 3,
                "time": "2018-09-15 20:39:58.999999",
                "sender_type": "用户",
                "send_content": "男的",
                "act_tag": "用户回答",
                "attributes": [
                    {
                        "key": "人群",
                        "value": "男",
                        "start": 0,
                        "exclusive_end": 1
                    }
                ],
                "rec_item_id": [],
                "rec_item_info": []
            },
            {
                "seq_no": 4,
                "time": "2018-09-15 20:40:01.999998",
                "sender_type": "用户",
                "send_content": "颜色白色",
                "act_tag": "用户回答",
                "attributes": [
                    {
                        "key": "颜色",
                        "value": "白色",
                        "start": 2,
                        "exclusive_end": 4
                    }
                ],
                "rec_item_id": [],
                "rec_item_info": []
            },
            {
                "seq_no": 5,
                "time": "2018-09-15 20:40:11",
                "sender_type": "用户",
                "send_content": "尺码41.5",
                "act_tag": "用户回答",
                "attributes": [
                    {
                        "key": "鞋码",
                        "value": "41.5",
                        "start": 2,
                        "exclusive_end": 6
                    }
                ],
                "rec_item_id": [],
                "rec_item_info": []
            },
            {
                "seq_no": 6,
                "time": "2018-09-15 20:40:42.000001",
                "sender_type": "客服",
                "send_content": "仅发送商品链接",
                "act_tag": "系统推荐",
                "attributes": [],
                "rec_item_id": [
                    "610616103106"
                ],
                "rec_item_info": [
                    "610616103106@50012029"
                ]
            },
            {
                "seq_no": 7,
                "time": "2018-09-15 20:40:43.999997",
                "sender_type": "客服",
                "send_content": "仅发送商品链接",
                "act_tag": "系统推荐",
                "attributes": [],
                "rec_item_id": [
                    "667276612374"
                ],
                "rec_item_info": [
                    "667276612374@50012029"
                ]
            },
            {
                "seq_no": 8,
                "time": "2018-09-15 20:40:46.000001",
                "sender_type": "客服",
                "send_content": "仅发送商品链接",
                "act_tag": "系统推荐",
                "attributes": [],
                "rec_item_id": [
                    "666161880918"
                ],
                "rec_item_info": [
                    "666161880918@50012029"
                ]
            },
            {
                "seq_no": 9,
                "time": "2018-09-15 20:41:07.999999",
                "sender_type": "用户",
                "send_content": "500左右的",
                "act_tag": "用户反馈",
                "attributes": [
                    {
                        "key": "价位",
                        "value": "500左右",
                        "start": 0,
                        "exclusive_end": 5
                    }
                ],
                "rec_item_id": [],
                "rec_item_info": []
            },
            {
                "seq_no": 10,
                "time": "2018-09-15 20:41:30.999998",
                "sender_type": "客服",
                "send_content": "仅发送商品链接",
                "act_tag": "系统推荐",
                "attributes": [],
                "rec_item_id": [
                    "674993136194"
                ],
                "rec_item_info": [
                    "674993136194@50012029"
                ]
            },
            {
                "seq_no": 11,
                "time": "2018-09-15 20:46:48",
                "sender_type": "用户",
                "send_content": "还有其他的嘛？",
                "act_tag": "用户反馈",
                "attributes": [],
                "rec_item_id": [],
                "rec_item_info": []
            },
            {
                "seq_no": 12,
                "time": "2018-09-15 20:47:32.999997",
                "sender_type": "客服",
                "send_content": "仅发送商品链接",
                "act_tag": "系统推荐",
                "attributes": [],
                "rec_item_id": [
                    "662003809511"
                ],
                "rec_item_info": [
                    "662003809511@50012029"
                ]
            },
            {
                "seq_no": 13,
                "time": "2018-09-15 20:47:52.999999",
                "sender_type": "客服",
                "send_content": "仅发送商品链接",
                "act_tag": "系统推荐",
                "attributes": [],
                "rec_item_id": [
                    "633205338921"
                ],
                "rec_item_info": [
                    "633205338921@50012029"
                ]
            },
            {
                "seq_no": 14,
                "time": "2018-09-15 20:51:24.999998",
                "sender_type": "用户",
                "send_content": "我喜欢老爹鞋",
                "act_tag": "用户反馈",
                "attributes": [
                    {
                        "key": "品类",
                        "value": "老爹鞋",
                        "start": 3,
                        "exclusive_end": 6
                    }
                ],
                "rec_item_id": [],
                "rec_item_info": []
            },
            {
                "seq_no": 15,
                "time": "2018-09-15 20:52:13.000003",
                "sender_type": "客服",
                "send_content": "仅发送商品链接",
                "act_tag": "系统推荐",
                "attributes": [],
                "rec_item_id": [
                    "627113411982"
                ],
                "rec_item_info": [
                    "627113411982@50012029"
                ]
            }
        ]
    }
```

# 大家电 Appliance
```json
{
        "domain": "大家电行业",
        "sid": "080aeb0d4302fb09dc0636e0b1d13cc4",
        "sellerid": "2616972755",
        "userid": "13795785",
        "ds": "20181005",
        "dialogue": [
            {
                "seq_no": 1,
                "time": "2018-10-05 18:49:31.999999",
                "sender_type": "用户",
                "send_content": "帮我推荐一款普通洗衣机，性价比高，皮实耐用的，不要烘干功能的",
                "act_tag": "用户需求",
                "attributes": [
                    {
                        "key": "品类",
                        "value": "洗衣机",
                        "start": 8,
                        "exclusive_end": 11
                    },
                    {
                        "key": "功能需求",
                        "value": "皮实耐用",
                        "start": 17,
                        "exclusive_end": 21
                    },
                    {
                        "key": "功能需求",
                        "value": "不要烘干",
                        "start": 23,
                        "exclusive_end": 27
                    },
                    {
                        "key": "款式",
                        "value": "性价比高",
                        "start": 12,
                        "exclusive_end": 16
                    }
                ],
                "rec_item_id": [],
                "rec_item_info": []
            },
            {
                "seq_no": 2,
                "time": "2018-10-05 18:49:57.000003",
                "sender_type": "客服",
                "send_content": "波轮还是滚筒呢",
                "act_tag": "系统提问",
                "attributes": [
                    {
                        "key": "洗衣机类型",
                        "value": "波轮",
                        "start": 0,
                        "exclusive_end": 2
                    },
                    {
                        "key": "洗衣机类型",
                        "value": "滚筒",
                        "start": 4,
                        "exclusive_end": 6
                    }
                ],
                "rec_item_id": [],
                "rec_item_info": []
            },
            {
                "seq_no": 3,
                "time": "2018-10-05 18:50:04.000001",
                "sender_type": "用户",
                "send_content": "滚筒的",
                "act_tag": "用户回答",
                "attributes": [
                    {
                        "key": "洗衣机类型",
                        "value": "滚筒",
                        "start": 0,
                        "exclusive_end": 2
                    }
                ],
                "rec_item_id": [],
                "rec_item_info": []
            },
            {
                "seq_no": 4,
                "time": "2018-10-05 18:50:28.000003",
                "sender_type": "用户",
                "send_content": "功能简单的",
                "act_tag": "用户回答",
                "attributes": [
                    {
                        "key": "功能需求",
                        "value": "功能简单",
                        "start": 0,
                        "exclusive_end": 4
                    }
                ],
                "rec_item_id": [],
                "rec_item_info": []
            },
            {
                "seq_no": 5,
                "time": "2018-10-05 18:51:11.999998",
                "sender_type": "客服",
                "send_content": "仅发送商品链接",
                "act_tag": "系统推荐",
                "attributes": [],
                "rec_item_id": [
                    "666583252553"
                ],
                "rec_item_info": [
                    "666583252553@50022703"
                ]
            },
            {
                "seq_no": 6,
                "time": "2018-10-05 18:51:18.999997",
                "sender_type": "客服",
                "send_content": "仅发送商品链接",
                "act_tag": "系统推荐",
                "attributes": [],
                "rec_item_id": [
                    "646804567416"
                ],
                "rec_item_info": [
                    "646804567416@50022703"
                ]
            },
            {
                "seq_no": 7,
                "time": "2018-10-05 18:51:49.999997",
                "sender_type": "用户",
                "send_content": "有A品牌的吗",
                "act_tag": "用户反馈",
                "attributes": [
                    {
                        "key": "品牌",
                        "value": "A品牌",
                        "start": 1,
                        "exclusive_end": 4
                    }
                ],
                "rec_item_id": [],
                "rec_item_info": []
            },
            {
                "seq_no": 8,
                "time": "2018-10-05 18:52:13.999999",
                "sender_type": "客服",
                "send_content": "这款洗衣机有5大功能：1是有专属净柔洗程序，如同手洗般轻柔、揉搓间为衣物重塑洁净与柔软；2是支持95度高温煮洗，扫净藏于衣物纤维中的病毒细菌，长效杀菌灭毒，99.9%健康除菌；3是可以wifi手机远程控制，随时随地，想穿就穿；4是支持特色羽绒服洗，分多段进水，洗涤节拍柔和，预防羽绒服漂浮水面或破损，洗护均匀，贴心呵护；5是具备BLDC变频电机，脱水更快更彻底，洁净少残留",
                "act_tag": "系统推荐",
                "attributes": [],
                "rec_item_id": [
                    "662907178657"
                ],
                "rec_item_info": [
                    "662907178657@50022703"
                ]
            },
            {
                "seq_no": 9,
                "time": "2018-10-05 18:53:06.000003",
                "sender_type": "用户",
                "send_content": "波轮的哪款性价比高？皮实耐用",
                "act_tag": "用户反馈",
                "attributes": [
                    {
                        "key": "款式",
                        "value": "性价比高",
                        "start": 5,
                        "exclusive_end": 9
                    },
                    {
                        "key": "洗衣机类型",
                        "value": "波轮",
                        "start": 0,
                        "exclusive_end": 2
                    },
                    {
                        "key": "功能需求",
                        "value": "皮实耐用",
                        "start": 10,
                        "exclusive_end": 14
                    }
                ],
                "rec_item_id": [],
                "rec_item_info": []
            },
            {
                "seq_no": 10,
                "time": "2018-10-05 18:53:24",
                "sender_type": "客服",
                "send_content": "预算多少呢亲",
                "act_tag": "系统提问",
                "attributes": [
                    {
                        "key": "价位",
                        "value": "",
                        "start": -1,
                        "exclusive_end": -1
                    }
                ],
                "rec_item_id": [],
                "rec_item_info": []
            },
            {
                "seq_no": 11,
                "time": "2018-10-05 18:53:47",
                "sender_type": "用户",
                "send_content": "3000之内的，要皮实耐用，功能简单的",
                "act_tag": "用户回答",
                "attributes": [
                    {
                        "key": "价位",
                        "value": "3000之内",
                        "start": 0,
                        "exclusive_end": 6
                    },
                    {
                        "key": "功能需求",
                        "value": "皮实耐用",
                        "start": 9,
                        "exclusive_end": 13
                    },
                    {
                        "key": "功能需求",
                        "value": "功能简单",
                        "start": 14,
                        "exclusive_end": 18
                    }
                ],
                "rec_item_id": [],
                "rec_item_info": []
            },
            {
                "seq_no": 12,
                "time": "2018-10-05 18:53:55",
                "sender_type": "用户",
                "send_content": "不需要烘干功能",
                "act_tag": "用户回答",
                "attributes": [
                    {
                        "key": "功能需求",
                        "value": "不需要烘干",
                        "start": 0,
                        "exclusive_end": 5
                    }
                ],
                "rec_item_id": [],
                "rec_item_info": []
            },
            {
                "seq_no": 13,
                "time": "2018-10-05 18:54:55.999999",
                "sender_type": "客服",
                "send_content": "这是一款10公斤变频波轮洗衣机，支持搅拌水流（打散衣物），冲浪水流（均匀洗涤），喷瀑水流（翻滚衣物），高于国标0.8洗净比标准，支持纳米银离子健康除菌，除菌率99.9%，遇水释放，任何洗衣程序都可杀菌，拥有立体无孔内桶，防止二次污染，节能省水，多维度水流有效杀死衣物螨虫，劲漂脱水排出螨虫尸体，给宝贝皮肤多一层防护。它还支持wifi手机远程控制，提前洗涤，回家晾晒，无需等待。",
                "act_tag": "系统推荐",
                "attributes": [],
                "rec_item_id": [
                    "655328647490"
                ],
                "rec_item_info": [
                    "655328647490@50022703"
                ]
            },
            {
                "seq_no": 14,
                "time": "2018-10-05 19:18:28",
                "sender_type": "用户",
                "send_content": "请问A品牌有8公斤的洗衣机吗？波轮的",
                "act_tag": "用户反馈",
                "attributes": [
                    {
                        "key": "品类",
                        "value": "洗衣机",
                        "start": 10,
                        "exclusive_end": 13
                    },
                    {
                        "key": "品牌",
                        "value": "A品牌",
                        "start": 2,
                        "exclusive_end": 5
                    },
                    {
                        "key": "洗涤重量",
                        "value": "8公斤",
                        "start": 6,
                        "exclusive_end": 9
                    },
                    {
                        "key": "洗衣机类型",
                        "value": "波轮",
                        "start": 15,
                        "exclusive_end": 17
                    }
                ],
                "rec_item_id": [],
                "rec_item_info": []
            },
            {
                "seq_no": 15,
                "time": "2018-10-05 19:18:49.999997",
                "sender_type": "客服",
                "send_content": "这是一款8公斤大容量变频波轮洗衣机,全新免清洗，直驱更静音，有3大特点：1是健康除螨洗护，多维度水流冲刷除螨，劲漂脱水排出螨虫，给家人皮肤多一层防护；2是8公斤大容量节能，大件洗涤统统搞定，全家衣物一桶搞定；3是0-9小时预约时间，剩余时间显示，您的时间您做主",
                "act_tag": "系统推荐",
                "attributes": [],
                "rec_item_id": [
                    "637474093533"
                ],
                "rec_item_info": [
                    "637474093533@50022703"
                ]
            }
        ]
    }
```
