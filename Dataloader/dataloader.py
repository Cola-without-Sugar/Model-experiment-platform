from msilib import sequence
import torch.utils.data as data
import os
import PIL.Image as Image

#data.Dataset:
#所有子类应该override__len__和__getitem__，前者提供了数据集的大小，后者支持整数索引，范围从0到len(self)

class LoadDataset(data.Dataset):
    # 创建读取分割数据集类的实例调用init
    def __init__(self, root:str ,transform = None , label_transform = None):
        self.imgs_root = root+"/imgs/"
        self.masks_root = root+"/masks/"
        n = len(os.listdir(self.imgs_root))
        
        sequence = []
        for i in range(n):
            img = self.imgs_root + os.listdir(self.imgs_root)[i]
            mask = self.masks_root + os.listdir(self.masks_root)[i]
            sequence.append([img,mask]) #append 只能有一个参数，加上[]变成list

        self.sequence = sequence
        self.transform = transform
        self.label_transform = label_transform
    
    #返回指定索引的图片
    def __getitem__(self, index: int):
        x_path,y_path = self.sequence[index]
        img_x = Image.open(x_path)
        mask_y = Image.open(y_path)
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.label_transform is not None:
            mask_y = self.label_transform(mask_y)
        return img_x,mask_y 

    #返回数据集中的图片和掩膜
    def __len__(self):
        return len(self.sequence)


    #判断数据集和标签数据的数量是否对的上，如果对得上返回True，对不上返回False
    def isavilable(self):
        n = len(os.listdir(self.imgs_root))
        m = len(os.listdir(self.masks_root))
        if (n==m):
            return True
        else:
            return False
