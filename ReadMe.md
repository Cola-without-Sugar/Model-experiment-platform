# 深度学习实验平台的搭建
这个文档主要关于深度学习实验平台的搭建的说明文件，主要是用于深度模型训练和超参数选择方向的训练使用的文档。目标是提供多种数据处理任务和多种模型的训练方式和结果的训练。

## 主要超参数解释
训练时使用命令行输入超参数的模式运行程序
想要复现论文需要对论文的数据集进行复现点数，即确定复现下的精度是否达到原论文或者在paperwithcode中报告的精度，并以此作为训练数据集的最终目标

## 扩展平台
### 添加模型
1. 首先确保使用的训练模型在程序的`model_list`中。并保证名称正确
2. 如果名称不正确，修改对应参数的名称
3. 如果不存在，则在BackBone文件夹下创建指定模型的网络结构，然后在`model_list`中添加指定模型，并导入模型架构

### 添加数据信息

## 复现笔记
### Unet
1. 在双向传输路径的过程中，最后的特征向量图无法与最终得到的模板匹配，是因为在传播的过程中图片降维除2取整，会有一些误差 