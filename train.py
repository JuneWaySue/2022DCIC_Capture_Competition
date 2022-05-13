import os
import gc
import string
import time

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import SGD,lr_scheduler,Adam
from sklearn.model_selection import KFold,StratifiedKFold
from torch.nn import CrossEntropyLoss,MultiLabelSoftMarginLoss,BCEWithLogitsLoss

try:
    import timm
except:
    ! pip install timm
    import timm
try:
    import aug_lib
except:
    ! wget https://github.com/automl/trivialaugment/raw/master/aug_lib.py
    import aug_lib

from utils import train_model,valid_model,train_model_4fc,valid_model_4fc
from dataset import ImageDataSet_fold
from models import TimmModels,TimmModels_4fc


def train():
    '''resnetrs50 1fc  5fold'''
    train_path='../input/2022dcic-capture/training_dataset/training_dataset/'
    test_path='../input/2022dcic-capture/test_dataset/test_dataset/'
    all_str = {v: k for k, v in enumerate(list(string.digits + string.ascii_letters))}
    captcha_number=4
    epochs = 20
    rate = 0.001
    batch_size=32
    img_width = 512
    img_height = 256
    pin_memory=True
    num_workers=os.cpu_count()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_transform = transforms.Compose([
        transforms.Resize((img_height,img_width)),
        transforms.RandomRotation((-5, 5)),
        aug_lib.TrivialAugment(),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((img_height,img_width)),
        transforms.ToTensor(),
    ])
    kf=KFold(n_splits=5,random_state=2022,shuffle=True)
    files=np.array([train_path+i for i in os.listdir(train_path)])
    for idx,(train_ind,valid_ind) in enumerate(kf.split(files)):
        print(f'start train kfold{idx+1}...')
        
        train_dataset=ImageDataSet_fold(files[train_ind].tolist(),train_transform)
        train_dataloader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,pin_memory=pin_memory)

        valid_dataset=ImageDataSet_fold(files[valid_ind].tolist(),test_transform)
        valid_dataloader=DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,pin_memory=pin_memory)
        
        model=TimmModels(model_name='resnetrs50',pretrained=True).to(device)
        criterion = MultiLabelSoftMarginLoss().to(device)
        optimizer = Adam(model.parameters(), lr=rate)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=len(train_dataloader) / 10, gamma=0.95)
        
        best_acc = 0
        best_epoch = 0
        for epoch in range(epochs):
            char_pred,char_true,train_loss,train_acc,train_time_cost = train_model(model,train_dataloader,criterion,optimizer,scheduler)
            valid_acc,valid_time_cost,valid_loss,char_df_valid = valid_model(model,valid_dataloader,criterion,flag=False)
            
            char_df=pd.DataFrame([char_true],index=['train_true']).T.reset_index().merge(pd.DataFrame([char_pred],index=['train_pred']).T.reset_index(),on='index',how='left').fillna(0)
            char_df['train_percent']=char_df['train_pred']/char_df['train_true']
            char_df['train_pred']=char_df['train_pred'].astype(int)
            char_df=char_df.merge(char_df_valid,on='index',how='left')
            char_df=char_df.sort_values(by='valid_percent',ascending=False).reset_index(drop=True)
            char_df['label']=char_df['index'].map(all_str)
            char_df.to_csv(f'char_df_kfold{idx+1}.csv',index=False)
            print(f"[kfold{idx+1} {epoch+1}/{epochs}], train's loss:{train_loss:.5f}, train's accuracy:{train_acc:.2%}, vaild's loss:{valid_loss:.5f}, vaild's accuracy:{valid_acc:.2%}, time:{train_time_cost+valid_time_cost} s")
            if valid_acc > best_acc:
                # 保存训练结果
                torch.save(model.state_dict(), f'resnetrs50_kfold{idx+1}.pth')
                best_acc=valid_acc
                best_epoch=epoch+1
        print(f'done train kfold{idx+1}...')
        print(f'best_epoch:{best_epoch},best_acc:{best_acc:.2%}')
        print('-'*50)
        del model,criterion,optimizer,scheduler,train_dataset,train_dataloader,valid_dataset,valid_dataloader
        gc.collect()
        print('清理内存成功！！')


def train_4fc():
    '''tf_efficientnet_b7 4fc  5fold'''
    train_path='../input/2022dcic-capture/training_dataset/training_dataset/'
    test_path='../input/2022dcic-capture/test_dataset/test_dataset/'
    all_str = {v: k for k, v in enumerate(list(string.digits + string.ascii_letters))}
    captcha_number=4
    epochs = 30
    rate = 0.001
    batch_size=10
    img_width = 512
    img_height = 256
    pin_memory=True
    num_workers=os.cpu_count()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_transform = transforms.Compose([
        transforms.Resize((img_height,img_width)),
        transforms.RandomRotation((-5, 5)),
        aug_lib.TrivialAugment(),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((img_height,img_width)),
        transforms.ToTensor(),
    ])
    kf=KFold(n_splits=5,random_state=2022,shuffle=True)
    files=np.array([train_path+i for i in os.listdir(train_path)])
    for idx,(train_ind,valid_ind) in enumerate(kf.split(files)):
        print(f'start train kfold{idx+1}...')
        
        train_dataset=ImageDataSet_fold(files[train_ind].tolist(),train_transform)
        train_dataloader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,pin_memory=pin_memory)

        valid_dataset=ImageDataSet_fold(files[valid_ind].tolist(),test_transform)
        valid_dataloader=DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,pin_memory=pin_memory)
        
        model=TimmModels_4fc(model_name='tf_efficientnet_b7',pretrained=True).to(device)
        criterion = CrossEntropyLoss().to(device)
        optimizer = Adam(model.parameters(), lr=rate)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=len(train_dataloader) // 10, gamma=0.99)
        
        best_acc = 0
        best_epoch = 0
        for epoch in range(epochs):
            char_pred,char_true,train_loss,train_acc,train_time_cost = train_model_4fc(model,train_dataloader,criterion,optimizer,scheduler)
            valid_acc,valid_time_cost,valid_loss,char_df_valid = valid_model_4fc(model,valid_dataloader,criterion)
            
            char_df=pd.DataFrame([char_true],index=['train_true']).T.reset_index().merge(pd.DataFrame([char_pred],index=['train_pred']).T.reset_index(),on='index',how='left').fillna(0)
            char_df['train_percent']=char_df['train_pred']/char_df['train_true']
            char_df['train_pred']=char_df['train_pred'].astype(int)
            char_df=char_df.merge(char_df_valid,on='index',how='left')
            char_df=char_df.sort_values(by='valid_percent',ascending=False).reset_index(drop=True)
            char_df['label']=char_df['index'].map(all_str)
            char_df.to_csv(f'char_df_{epoch+1}.csv',index=False)
            print(f"[kfold{idx+1} {epoch+1}/{epochs}], train's loss:{train_loss:.5f}, train's accuracy:{train_acc:.2%}, vaild's loss:{valid_loss:.5f}, vaild's accuracy:{valid_acc:.2%}, time:{train_time_cost+valid_time_cost} s")
            if valid_acc > best_acc:
                # 保存训练结果
                torch.save(model.state_dict(), f'tf_efficientnet_b7_kfold{idx+1}.pth')
                best_acc=valid_acc
                best_epoch=epoch+1
        print(f'done train kfold{idx+1}...')
        print(f'best_epoch:{best_epoch},best_acc:{best_acc:.2%}')
        print('-'*50)
        del model,criterion,optimizer,scheduler,train_dataset,train_dataloader,valid_dataset,valid_dataloader
        gc.collect()
        print('清理内存成功！！')

def train_all():
    '''tf_efficientnet_b7 4fc  all'''
    train_path='../input/2022dcic-capture/training_dataset/training_dataset/'
    test_path='../input/2022dcic-capture/test_dataset/test_dataset/'
    all_str = {v: k for k, v in enumerate(list(string.digits + string.ascii_letters))}
    captcha_number=4
    epochs = 38
    rate = 0.001
    batch_size=10
    img_width = 512
    img_height = 256
    pin_memory=True
    num_workers=os.cpu_count()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_transform = transforms.Compose([
        transforms.Resize((img_height,img_width)),
        transforms.RandomRotation((-5, 5)),
        aug_lib.TrivialAugment(),
        transforms.ToTensor(),
    ])

    files=[train_path+i for i in os.listdir(train_path)]

    train_dataset=ImageDataSet_fold(files,train_transform)
    train_dataloader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,pin_memory=pin_memory)

    model=TimmModels_4fc(model_name='tf_efficientnet_b7',pretrained=True).to(device)
    criterion = CrossEntropyLoss().to(device)
    optimizer = Adam(model.parameters(), lr=rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=len(train_dataloader) // 10, gamma=0.99)

    best_acc = 0
    best_acc_epoch = 0
    best_loss = 10000
    best_loss_epoch = 0

    print(f'start train...')
    print(f'train_dataloader iters is {len(train_dataloader)}')
    for epoch in range(epochs):
        char_pred,char_true,train_loss,train_acc,train_time_cost = train_model_4fc(model,train_dataloader,criterion,optimizer,scheduler)

        char_df=pd.DataFrame([char_true],index=['train_true']).T.reset_index().merge(pd.DataFrame([char_pred],index=['train_pred']).T.reset_index(),on='index',how='left').fillna(0)
        char_df['train_percent']=char_df['train_pred']/char_df['train_true']
        char_df['train_pred']=char_df['train_pred'].astype(int)
        char_df['label']=char_df['index'].map(all_str)
        char_df.to_csv(f'char_df_{epoch+1}.csv',index=False)
        print(f"[{epoch+1}/{epochs}], train's loss:{train_loss:.5f}, train's accuracy:{train_acc:.2%}, time:{train_time_cost} s")
        if train_acc > best_acc:
            # 保存训练结果
            torch.save(model.state_dict(), 'train_acc_best.pth')
            best_acc=train_acc
            best_acc_epoch=epoch+1
        if train_loss < best_loss:
            # 保存训练结果
            torch.save(model.state_dict(), 'train_loss_best.pth')
            best_loss=train_loss
            best_loss_epoch=epoch+1
    torch.save(model.state_dict(), 'the_last_epoch.pth')
    print(f'done train...')
    print(f'best_acc_epoch:{best_acc_epoch},best_acc:{best_acc:.2%}')
    print(f'best_loss_epoch:{best_loss_epoch},best_loss:{best_loss:.5f}')


if __name__ == '__main__':
    train()

    train_4fc()

    train_all()