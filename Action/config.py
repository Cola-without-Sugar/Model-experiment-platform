### 参数定义文件
from Dataloader.dataloader import LoadDataset
from Backbone.Unet import UNet
from torchvision.transforms import transforms,InterpolationMode
import torch

#数据集加载的地址路径还有一些别的超参数
base_root = r"D:\\REALLY_WANT_TO_Learning\\github_repositories\\0-Model实验平台\\01-Dataset\\"
dataset_name = r"ChaseDB-1"
result_dir = r"D:/REALLY_WANT_TO_Learning/github_repositories/0-Model实验平台/02-Result/"
result_dir = result_dir + "weight/"+dataset_name

#数据集输入的处理过程 将图片放缩至572*572，与论文中的保持一致，同时减少显存的占用
input_transform = transforms.Compose([
    transforms.Resize([512,512]),
    transforms.ToTensor(),
    # 标准化至[-1,1],规定均值和标准差
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    #torchvision.transforms.Normalize(mean, std, inplace=False)
])

# mask只需要转换为tensor,在进行数据的调整过程中，resize函数默认使用的是双线性插值法，而蒙版中只有0,1两个标签，所以修改为最近邻插值
mask_transform = transforms.Compose([
                 transforms.Resize([512,512],interpolation = InterpolationMode.NEAREST),
                 transforms.ToTensor()])

# 显示已经实现的模型列表
model_list = {'unet':UNet}


def interpret(opts):
    ##main选项的opts进行初始化
    
    # 准备灵材
    # 根据不同任务选择不同的数据集加载模式 
    # segmentation任务-数据集包括图像与mask
    if (opts.task == 'segmentation'):
        print("分割读取程序开始")
        dataset = LoadDataset(base_root+dataset_name,input_transform,mask_transform)
    # 图像分类任务-数据集包括图像与分类标签
    elif (opts.task == 'classification'):
        pass 
    # 图像检测任务-数据集包括图像与定位信息标签
    elif (opts.task == 'detection'):
        pass

    # 准备丹炉，目前是单机单卡模式，以后再添加单机多卡模式或多机多卡模式
    if (opts.device == 'cpu'):
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 准备灵阵（模型）
    if (opts.model in model_list):
        model = model_list[opts.model]
        batch_size = opts.batch_size
    else:
        print('model不在可实现的队列中，请参考ReadMe添加')
    
    return model,dataset,device,batch_size