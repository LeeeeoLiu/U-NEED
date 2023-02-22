# 本仓库实现了任务2的两种baseline
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