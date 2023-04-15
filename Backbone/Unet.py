import torch
import torchvision.transforms.functional
from torch import nn

#首先参考论文数据进行搭建，由于论文常常是两层3*3卷积所以定义如下
#原论文中使用的是0padding，这里使用1padding
class DoubleConvolution(nn.Module):
    def __init__(self, in_channels:int, out_channels:int):
        super().__init__()
        self.first = nn.Conv2d(in_channels, out_channels , kernel_size = 3 ,padding =1)
        self.act1 = nn.ReLU()
        self.second = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

    def forward(self,x:torch.Tensor):
        x = self.first(x)
        x = self.act1(x)
        x = self.second(x)
        return self.act2(x)

# 下采样中使用每一层都用了2*2的最大池化层
class DownSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

    def forward(self,x: torch.Tensor):
        return self.pool(x)

# 上采样中使用的是2*2的转置卷积，也就是反卷积
class UpSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels,out_channels,kernel_size = 2,stride = 2)
    
    def forward(self,x:torch.Tensor):
        return self.up(x)

# 裁剪和拼接特征图 x:当前在扩张路径中的特征图， contracting_x：对应在压缩路径中的特征图
class CropAndConcat(nn.Module):
    def forward(self,x:torch.Tensor,contracting_x:torch.Tensor):
        contracting_x = torchvision.transforms.functional.center_crop(contracting_x, [x.shape[2], x.shape[3]])
        x = torch.cat([x, contracting_x], dim = 1)
        return x

##################
##   U-net      ##
##################
class UNet(nn.Module):
    def __init__(self, in_channels:int,out_channels:int):
        super().__init__()
        self.down_conv = nn.ModuleList([DoubleConvolution(i,o) for i,o in
                                        [(in_channels,64),(64,128),(128,256),(256,512)]])
        
        self.down_sample = nn.ModuleList([DownSample() for _ in range(4)])

        self.middle_conv = DoubleConvolution(512,1024)

        self.up_sample = nn.ModuleList([UpSample(i,o) for i,o in 
                                        [(1024,512),(512,256),(256,128),(128,64)]])

        self.up_conv = nn.ModuleList([DoubleConvolution(i,o) for i,o in 
                                        [(1024,512),(512,256),(256,128),(128,64)]])

        self.concat = nn.ModuleList([CropAndConcat() for _ in range(4)])

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size = 1)

    def forward(self,x:torch.Tensor):
        pass_through = []

        for i in range(len(self.down_conv)):
            x = self.down_conv[i](x)
            pass_through.append(x)
            x = self.down_sample[i](x)
        
        x = self.middle_conv(x)

        for i in range(len(self.up_conv)):
            x = self.up_sample[i](x)
            x = self.concat[i](x,pass_through.pop())
            x = self.up_conv[i](x)
        
        x = self.final_conv(x)

        return x