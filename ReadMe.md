# 深度学习实验平台的搭建
这个文档主要关于深度学习实验平台的搭建的说明文件，主要是用于深度模型训练和超参数选择方向的训练使用的文档。目标是提供多种数据处理任务和多种模型的训练方式和结果的训练。

## 主要超参数解释
训练时使用命令行输入超参数的模式运行程序
想要复现论文需要对论文的数据集进行复现点数，即确定复现下的精度是否达到原论文或者在paperwithcode中报告的精度，并以此作为训练数据集的最终目标

## 使用平台实验指南
### 添加模型
1. 首先确保使用的训练模型在程序的`model_list`中。并保证名称正确
2. 如果名称不正确，修改对应参数的名称
3. 如果不存在，则在BackBone文件夹下创建指定模型的网络结构，然后在`model_list`中添加指定模型，并导入模型架构

### 添加数据信息
数据集的数据位置路径应当在`Action\config`下进行修改，保证数据集与对应任务的分配保持一致。

## 附录

### 复现实验笔记
### 创新实验笔记

### 更新日志
2023-4-15 ：上传更新了实验平台，主要调整基础模型的输入数据与输出结果权重文件的定义命名方式

### 待解决意见及问题
1. 需要优化结果的表现形式和表现内容，增加损失和精度计算函数和图表函数
2. 可以使用tensoflow的那个管理run文件来显示训练内容
3. Unet复现的实验精度还未计算，需要摸清测试状态下精度是否达到报告的精度