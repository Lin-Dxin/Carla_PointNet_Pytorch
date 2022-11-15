# 进行KFLOD-并行实验的须知

## 概述

该实验主要包括以下步骤：

- 数据集预处理
- 修改train_carla_semseg.py的信息
- 生成初始化模型以及运行代码

## 数据集预处理

### 将数据划分为Test以及TrainAndValidate两部分

- 首先将所有数据放置同一个文件夹内，也将divide_train_test.py放入文件夹中
- 在cmd或者anaconda中打开该文件夹，输入：`python divide_train_test.py`
- 会提示该操作将移动该目录下的文件，输入 `y` 回车
- 执行完毕后将会将数据按 **8:2** 的比例划分至**TrainAndValidate以及TestData文件**下

### 将TrainAndValidate中的数据处理为对应的part

- 接下来将按照10折交叉验证进行划分，**需要生成 0 ~ 9 十个part的TrainAndValidate数据**
- 操作前**复制多份TrainAndValidate文件夹**，对其**重命名为**TrainAndValidate_0、TrainAndValidate_1 ..... TrainAndValidate_9
- 在复制期间，可以将已经复制完成并且重命名完成的文件夹进行进一步划分（划分训练集以及验证集），以下以TrainAndValidate_0为例
- 将**make_kflod_files.py放入TrainAndValidate_0文件夹**中
- 在cmd或者anaconda中打开该文件夹，输入：`python make_kflod_files.py`
- 会提示该操作将移动该目录下的文件，输入 `y` 回车
- 会再次提示确定该次操作的分区数，**输入(0-9)数字以确定分区**
- 回车后，操作结束

## 修改train_carla_semseg.py的信息

- 打开`train_carla_semseg.py`，修改对应的partition（位于30行）
- 修改modelinfo（35行），用于后续方便记录

## 生成初始化模型以及运行代码

- KFLOD需要保证同一组实验内的初始模型相同，因此需要准备一个初始化模型
- 修改`train_carla_semseg.py`中 SAVE_INIT 为True
- 在对应的log目录下找到`initial_state.pt`,修改名字放置`Pointnet_Pointnet2_pytorch`文件下
- 修改`train_carla_semseg.py`中的`model_path`与上一步保存的初始化模型名字相同
- 运行代码
- 运行后检查log目录下**是否生成对应model_info的log文件**，若有则正常，若无请检查
- 当模型跑完后，会生成`evaluation_data0.pth`
- 只需要将同一组的所有`evaluation_data.pth`放置同一个目录修改`4d_pn2_evaluate`
- 使用写好的可视化的代码`read_data.ipynb`去查看结果