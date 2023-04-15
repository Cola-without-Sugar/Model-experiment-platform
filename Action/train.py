import torch
from torch import optim
from torch.utils.data import DataLoader

class Segtrain():
    #初始化训练过程中的参数，目前学习率因为是阶梯型变化，所以无法确定，其余想要修改可以根据命令行修改
    def __init__(self,model,dataset,device,batch_size):
        self.model = model
        self.dataset = dataset
        self.device = device
        #定义损失函数，二分类函数，以及梯度优化函数
        self.lossfunction = torch.nn.BCELoss()
        self.sigmoid = torch.nn.Sigmoid()
        self.optimizer = optim.Adam
        #一些超参数的定义
        self.image_channels : int = 3
        self.mask_channels : int = 1
        self.batch_size = batch_size
        self.learning_rate: float = 2.5e-4
        self.epochs: int = 20

    def train(self):
        model = self.model(self.image_channels,self.mask_channels).to(self.device)
        #损失函数
        lossfunction = self.lossfunction
        #梯度下降优化器
        optimizer = self.optimizer(model.parameters())

        #加载数据集
        dataloader = DataLoader(self.dataset,batch_size = self.batch_size,shuffle = True,num_workers=0)
        self.train_model(model,lossfunction,optimizer,dataloader)

    def train_model(self,model,lossfunction,optimizer,dataloader):
        for epoch in range (self.epochs):
            print('Epoch {}/{}'.format(epoch, self.epochs - 1))
            print('-' * 10)
            dataset_size = len(dataloader.dataset)
            epoch_loss = 0
            step = 0  # minibatch数
            for x,y in dataloader:
                optimizer.zero_grad()
                inputs = x.to(self.device)
                labels = y.to(self.device)
                outputs = model(inputs)
                loss = lossfunction(self.sigmoid(outputs),labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                step+=1
            print("%d/%d,train_loss:%0.3f" % (step, dataset_size // dataloader.batch_size, loss.item()))
        torch.save(model.state_dict(),"Result\weight\weights_%d.pth" % epoch)
        return model
        
        #里面简单将训练模型得到的参数根据每一轮次保存了下来