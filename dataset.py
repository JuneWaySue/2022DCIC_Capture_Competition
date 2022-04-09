from torch.utils.data import Dataset
from PIL import Image
import torch
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

from utils import one_hot_encode,label_transform

class ImageDataSet(Dataset):
    '''图片加载和处理'''
    
    def __init__(self,file_path,transform=None,label_transform=None,is_train=True,is_valid=True,in_channels=3):
        self.files=[file_path+i for i in os.listdir(file_path)]
#         self.files=[i for i in self.files if i not in untrain_path]
        self.is_train = is_train
        if is_train:
            train,valid=train_test_split(self.files,test_size=0.2,random_state=2022,shuffle=True)
            if is_valid:
                self.files=valid
            else:
                self.files=train
        self.transform=transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = Path(self.files[idx])
        stem_list = list(image_path.stem)
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image=self.transform(image)
        if not self.is_train:
            return image,image_path.stem
        vector, order = one_hot_encode(stem_list)
        label = torch.from_numpy(vector)
        if label_transform is not None:
            label = label_transform(label,image_path.stem)
        return image, label, order, str(image_path)
    
class ImageDataSet_fold(Dataset):
    '''图片加载和处理'''
    
    def __init__(self,files,transform=None):
        self.files=files
        self.transform=transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = Path(self.files[idx])
        stem_list = list(image_path.stem)
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image=self.transform(image)
        vector, order = one_hot_encode(stem_list)
        label = torch.from_numpy(vector)
        return image, label, order, str(image_path)