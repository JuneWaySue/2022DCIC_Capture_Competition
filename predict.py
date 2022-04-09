import os
import string
from collections import defaultdict

import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable

from utils import one_hot_decode
from dataset import ImageDataSet
from models import TimmModels,TimmModels_4fc


def main():
    all_str = {v: k for k, v in enumerate(list(string.digits + string.ascii_letters))}
    # test_path='../input/2022dcic-capture/B_dataset/B榜/'
    test_path='../input/2022dcic-capture/test_dataset/test_dataset/'
    batch_size=128
    img_width = 512
    img_height = 256
    num_workers=os.cpu_count()
    test_transform = transforms.Compose([
        transforms.Resize((img_height,img_width)),
        transforms.ToTensor(),
    ])
    test_dataset=ImageDataSet(test_path,test_transform,is_train=False)
    test_dataloader=DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # resnetrs50 1fc 5fold 加权融合
    data=defaultdict(int)
    acc_map={1:0.9404,2:0.9461,3:0.9394,4:0.9314,5:0.9415}
    with torch.no_grad():
        for idx in range(5):
            model=TimmModels(model_name='resnetrs50',pretrained=False).to(device)
            model.load_state_dict(torch.load(f'../input/2022dcic-capture/5fold/resnetrs50_kfold{idx+1}.pth'))
            model.eval()
            for i, (images,stem) in tqdm(enumerate(test_dataloader),total=len(test_dataloader),desc=f'resnetrs50_kfold{idx+1}'):
                images = Variable(images).to(device)
                predict_labels = model(images / 255).detach()
                for each in range(len(stem)):
                    data[stem[each]]+=predict_labels[each] * acc_map[idx+1]
                    
    submit=[]
    for num,pred in tqdm(data.items()):
        p=pred / sum(acc_map.values())
        predict=one_hot_decode(p.view(4,62).max(1)[1])
        submit.append({'num':int(num),'tag':predict})
    df_resnet_5fold=pd.DataFrame(submit)

    # tf_efficientnet_b7 4fc 5fold 加权融合
    data=defaultdict(int)
    acc_map={1:0.9608,2:0.9624,3:0.9565,4:0.96,5:0.9607}
    with torch.no_grad():
        for idx in range(5):
            model=TimmModels_4fc(model_name='tf_efficientnet_b7',pretrained=False).to(device)
            model.load_state_dict(torch.load(f'../input/2022dcic-capture/b7_5fold/tf_efficientnet_b7_kfold{idx+1}.pth'))
            model.eval()
            for i, (images,stem) in tqdm(enumerate(test_dataloader),total=len(test_dataloader),desc=f'tf_efficientnet_b7_kfold{idx+1}'):
                images = Variable(images).to(device)
                pred1,pred2,pred3,pred4=model(images / 255)
                predict=torch.hstack((pred1,pred2,pred3,pred4))
                for each in range(len(stem)):
                    data[stem[each]]+=predict[each] * acc_map[idx+1]

    submit=[]
    for num,pred in tqdm(data.items()):
        p=pred / sum(acc_map.values())
        predict=one_hot_decode(p.view(4,62).max(1)[1])
        submit.append({'num':int(num),'tag':predict})
    df_b7_5fold=pd.DataFrame(submit)

    # best_acc_epoch:33,best_acc:98.71%
    # best_loss_epoch:37,best_loss:0.06261
    # the_last_epoch:38
    # all train 算术平均融合
    data=defaultdict(int)
    with torch.no_grad():
        for name in ['train_acc_best','train_loss_best','the_last_epoch']:
            model=TimmModels_4fc(model_name='tf_efficientnet_b7',pretrained=False).to(device)
            model.load_state_dict(torch.load(f'../input/2022dcic-capture/b7_all_train/{name}.pth'))
            model.eval()
            for i, (images,stem) in tqdm(enumerate(test_dataloader),total=len(test_dataloader),desc=name):
                images = Variable(images).to(device)
                pred1,pred2,pred3,pred4=model(images / 255)
                predict=torch.hstack((pred1,pred2,pred3,pred4))
                for each in range(len(stem)):
                    data[stem[each]]+=predict[each]

    submit=[]
    for num,pred in tqdm(data.items()):
        p=pred / 3
        predict=one_hot_decode(p.view(4,62).max(1)[1])
        submit.append({'num':int(num),'tag':predict})
    df_b7_all_train=pd.DataFrame(submit)

    # 最后，三个结果单字符投票，得出最终的结果
    def ronghe(x):
        tag_resnet_5folds=x['tag_resnet_5fold']
        tag_b7_5folds=x['tag_b7_5fold']
        tag_b7_all_trains=x['tag_b7_all_train']
        pred=''
        for i in range(4):
            p=defaultdict(int)
            tag_resnet_5fold = tag_resnet_5folds[i]
            tag_b7_5fold = tag_b7_5folds[i]
            tag_b7_all_train = tag_b7_all_trains[i]
            p[tag_resnet_5fold]+=1
            p[tag_b7_5fold]+=1
            p[tag_b7_all_train]+=1
            sort_p=sorted(p.items(),key=lambda x:x[1],reverse=True)
            if len(sort_p) <= 2:
                pred+=sort_p[0][0]
            else:
                # 三个都不一样的话用<tf_efficientnet_b7 4fc 5fold 加权融合>的结果
                pred+=tag_b7_5fold
        return pred

    df=(
        df_resnet_5fold.rename(columns={'tag':'tag_resnet_5fold'})
        .merge(df_b7_5fold.rename(columns={'tag':'tag_b7_5fold'}),on='num',how='left')
        .merge(df_b7_all_train.rename(columns={'tag':'tag_b7_all_train'}),on='num',how='left')
    )
    df['tag']=df.apply(ronghe,axis=1)
    df[['num','tag']].sort_values(by='num').reset_index(drop=True).to_csv('submit.csv',index=False)

if __name__ == '__main__':
    main()