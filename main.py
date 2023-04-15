#训练网络需要首先载入数据集
#加载模型，并进行初始化指定训练设备
#开始训练一个批次并保存数据
#在一定轮次后保存权重文件至指定文件

import argparse #argparse模块的作用是用于解析命令行参数 python parseTest.py input.txt --port=8080
from torch import optim

from Action.config import interpret
from Action.train import Segtrain
from Action.test import test


if __name__ == '__main__':
    # 参数解析 - 参考yolov5 
    parser = argparse.ArgumentParser()   #创建一个ArgumentParser对象
    parser.add_argument('action', type = str, help = 'train or test')
    parser.add_argument('model', type = str, help = 'deep learning model')
    parser.add_argument('task', type = str, help = 'the task to finish, classfication,detection or anyother')
    parser.add_argument('--device', type = str, default='', help = 'cpu or cuda:x')
    parser.add_argument('--batch_size', type = int, default = 4)
    parser.add_argument('--weight', type = str, help = 'the path of the mode weight file')
    opts = parser.parse_args()

    model,dataset,device,batch_size = interpret(opts)
    

    if opts.action == 'train':
        Segtrain(model,dataset,device,batch_size).train()
    elif opts.action == 'test':
        test()