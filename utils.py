from PIL import Image
import torch
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import cv2
import time
from collections import defaultdict
from torch.autograd import Variable
from math import floor
import random


def one_hot_encode(value: list) -> tuple:
    '''编码：将字符转为独热码，vector为独热码，order用于解码'''
    order = []
    shape = captcha_number * len(all_str)
    vector = np.zeros(shape, dtype=float)
    for k, v in enumerate(value):
        index = k * len(all_str) + all_str.get(v)
        vector[index] = 1.0
        order.append(all_str.get(v))
    return vector, order

def one_hot_decode(value: list) -> str:
    '''解码：将独热码转为字符'''
    res = []
    for ik, iv in enumerate(value):
        val = iv
        for k, v in all_str.items():
            if val == int(v):
                res.append(k)
                break
    return ''.join(res)

def print_img_mean_std(img_h=256,img_w=256):
    img_filenames = [train_path+i for i in os.listdir(train_path)]
    img_filenames += [test_path+i for i in os.listdir(test_path)]
    m_list, s_list = [], []
    for img_filename in tqdm(img_filenames):
        img=Image.open(img_filename).convert('RGB')
        img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (img_h, img_w))
        img = img / 255.0
        m, s = cv2.meanStdDev(img)
        m_list.append(m.reshape((3,)))
        s_list.append(s.reshape((3,)))
    m_array = np.array(m_list)
    s_array = np.array(s_list)
    m = m_array.mean(axis=0, keepdims=True)
    s = s_array.mean(axis=0, keepdims=True)
    print('均值：',m[0][::-1])
    print('标准差：',s[0][::-1])

def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    device = x.device
    batch_size = x.size()[0]

    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def valid_model(model,valid_dataloader,criterion,flag=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    s=time.time()
    model.eval()
    char_pred=defaultdict(int)
    char_true=defaultdict(int)
    correct, total  = 0, 0
    total_loss = 0
    with torch.no_grad():
        for i, (images, labels, orders, _) in enumerate(valid_dataloader):
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)
            orders = torch.vstack((orders[0],orders[1],orders[2],orders[3])).T
            predict_labels = model(images / 255)
            loss = criterion(predict_labels,labels)
            pred=predict_labels.view(-1,4,62).max(2)[1].cpu()
            equal=torch.eq(pred,orders)
            for char in one_hot_decode(orders[equal]):
                char_pred[char]+=1
            for each in orders:
                for char in one_hot_decode(each):
                    char_true[char]+=1
            total_loss += loss.item()
            for j,predict_label in enumerate(pred):
                prediction=predict_label
                predict=one_hot_decode(prediction)
                true=one_hot_decode(orders[j])
                total += 1
                if predict == true:
                    correct += 1
                else:
                    if flag:
                        print('Fail, capture:%s->%s' % (true, predict))
    char_df=pd.DataFrame([char_true],index=['valid_true']).T.reset_index().merge(pd.DataFrame([char_pred],index=['valid_pred']).T.reset_index(),on='index',how='left').fillna(0)
    char_df['valid_percent']=char_df['valid_pred']/char_df['valid_true']
    char_df['valid_pred']=char_df['valid_pred'].astype(int)
    char_df=char_df.sort_values(by='valid_percent',ascending=False).reset_index(drop=True)
    if flag:
        print(f'完成。总预测图片数为{total}张，准确率为{int(100 * correct / total)}%，loss为{total_loss/len(valid_dataloader):.5f}，耗时{int(time.time()-s)}s')
    else:
        return correct / total, int(time.time()-s), total_loss/len(valid_dataloader), char_df
    
def train_model(model,train_dataloader,criterion,optimizer,scheduler,mixup=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.train()
    total_loss = 0
    total=0
    correct=0
    char_pred=defaultdict(int)
    char_true=defaultdict(int)
    s=time.time()

    for i, (images, labels, orders, _) in enumerate(train_dataloader):
        if mixup:
            images, labels_a, labels_b, lam = mixup_data(images, labels, alpha)
            images = Variable(images).to(device)
            labels_a = Variable(labels_a).to(device)
            labels_b = Variable(labels_b).to(device)
        else:
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)
        predict_labels = model(images / 255)
        orders = torch.vstack((orders[0],orders[1],orders[2],orders[3])).T
        orders = Variable(orders).to(device)

        total += len(orders)
        pred=predict_labels.view(-1,4,62).max(2)[1].cpu()
        true=orders.cpu()
        equal=torch.eq(pred,true)
        for char in one_hot_decode(true[equal]):
            char_pred[char]+=1
        for each in true:
            for char in one_hot_decode(each):
                char_true[char]+=1
        correct += (equal.sum(1) == 4).sum().item()
        if mixup:
            loss = mixup_criterion(criterion, predict_labels, labels_a, labels_b, lam)
        else:
            loss = criterion(predict_labels,labels)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    return char_pred,char_true,total_loss / len(train_dataloader),correct/total,int(time.time()-s)

def train_model_4fc(model,train_dataloader,criterion,optimizer,scheduler=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.train()
    total_loss = 0
    total=0
    correct=0
    char_pred=defaultdict(int)
    char_true=defaultdict(int)
    s=time.time()
    for i, (images, labels, orders, _) in enumerate(train_dataloader):
        images = Variable(images).to(device)
        lable1,lable2,lable3,lable4 = orders
        lable1 = Variable(lable1).to(device)
        lable2 = Variable(lable2).to(device)
        lable3 = Variable(lable3).to(device)
        lable4 = Variable(lable4).to(device)

        pred1,pred2,pred3,pred4=model(images / 255)
        equal1=torch.eq(pred1.max(1)[1],lable1)
        equal2=torch.eq(pred2.max(1)[1],lable2)
        equal3=torch.eq(pred3.max(1)[1],lable3)
        equal4=torch.eq(pred4.max(1)[1],lable4)

        for char in one_hot_decode(lable1[equal1]):
            char_pred[char]+=1
        for char in one_hot_decode(lable1):
            char_true[char]+=1
        for char in one_hot_decode(lable2[equal2]):
            char_pred[char]+=1
        for char in one_hot_decode(lable2):
            char_true[char]+=1
        for char in one_hot_decode(lable3[equal3]):
            char_pred[char]+=1
        for char in one_hot_decode(lable3):
            char_true[char]+=1
        for char in one_hot_decode(lable4[equal4]):
            char_pred[char]+=1
        for char in one_hot_decode(lable4):
            char_true[char]+=1

        total += len(images)
        correct += (torch.vstack((equal1,equal2,equal3,equal4)).T.all(1).sum()).item()

        loss1,loss2,loss3,loss4 = criterion(pred1,lable1),criterion(pred2,lable2),criterion(pred3,lable3),criterion(pred4,lable4)
        loss = loss1+loss2+loss3+loss4
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

    return char_pred,char_true,total_loss / len(train_dataloader),correct/total,int(time.time()-s)

def valid_model_4fc(model,valid_dataloader,criterion):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    s=time.time()
    model.eval()
    char_pred=defaultdict(int)
    char_true=defaultdict(int)
    correct, total  = 0, 0
    total_loss = 0
    with torch.no_grad():
        for i, (images, labels, orders, _) in enumerate(valid_dataloader):
            images = Variable(images).to(device)
            lable1,lable2,lable3,lable4 = orders
            lable1 = Variable(lable1).to(device)
            lable2 = Variable(lable2).to(device)
            lable3 = Variable(lable3).to(device)
            lable4 = Variable(lable4).to(device)
            
            pred1,pred2,pred3,pred4=model(images / 255)
            equal1=torch.eq(pred1.max(1)[1],lable1)
            equal2=torch.eq(pred2.max(1)[1],lable2)
            equal3=torch.eq(pred3.max(1)[1],lable3)
            equal4=torch.eq(pred4.max(1)[1],lable4)

            for char in one_hot_decode(lable1[equal1]):
                char_pred[char]+=1
            for char in one_hot_decode(lable1):
                char_true[char]+=1
            for char in one_hot_decode(lable2[equal2]):
                char_pred[char]+=1
            for char in one_hot_decode(lable2):
                char_true[char]+=1
            for char in one_hot_decode(lable3[equal3]):
                char_pred[char]+=1
            for char in one_hot_decode(lable3):
                char_true[char]+=1
            for char in one_hot_decode(lable4[equal4]):
                char_pred[char]+=1
            for char in one_hot_decode(lable4):
                char_true[char]+=1
                
            total += len(images)
            correct += (torch.vstack((equal1,equal2,equal3,equal4)).T.all(1).sum()).item()
            
            loss1,loss2,loss3,loss4 = criterion(pred1,lable1),criterion(pred2,lable2),criterion(pred3,lable3),criterion(pred4,lable4)
            loss = loss1+loss2+loss3+loss4
            total_loss += loss.item()
            
    char_df=pd.DataFrame([char_true],index=['valid_true']).T.reset_index().merge(pd.DataFrame([char_pred],index=['valid_pred']).T.reset_index(),on='index',how='left').fillna(0)
    char_df['valid_percent']=char_df['valid_pred']/char_df['valid_true']
    char_df['valid_pred']=char_df['valid_pred'].astype(int)
    char_df=char_df.sort_values(by='valid_percent',ascending=False).reset_index(drop=True)
    return correct / total, int(time.time()-s), total_loss/len(valid_dataloader), char_df

class MyTransform():
    
    def __init__(self,p):
        self.p=p
    
    def __call__(self,image):
        if torch.rand(1) > self.p:
            return image
        grid_width=random.randint(2,10)
        grid_height=2
        magnitude=random.randint(1,10)
        image=self.perform_operation(image,grid_width,grid_height,magnitude)
        return image

    def perform_operation(self,image,grid_width,grid_height,magnitude):
        w, h = image.size

        horizontal_tiles = grid_width
        vertical_tiles = grid_height

        width_of_square = int(floor(w / float(horizontal_tiles)))
        height_of_square = int(floor(h / float(vertical_tiles)))

        width_of_last_square = w - (width_of_square * (horizontal_tiles - 1))
        height_of_last_square = h - (height_of_square * (vertical_tiles - 1))

        dimensions = []

        for vertical_tile in range(vertical_tiles):
            for horizontal_tile in range(horizontal_tiles):
                if vertical_tile == (vertical_tiles - 1) and horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square + (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif vertical_tile == (vertical_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square + (horizontal_tile * width_of_square),
                                       height_of_last_square + (height_of_square * vertical_tile)])
                elif horizontal_tile == (horizontal_tiles - 1):
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_last_square + (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])
                else:
                    dimensions.append([horizontal_tile * width_of_square,
                                       vertical_tile * height_of_square,
                                       width_of_square + (horizontal_tile * width_of_square),
                                       height_of_square + (height_of_square * vertical_tile)])
        last_column = []
        for i in range(vertical_tiles):
            last_column.append((horizontal_tiles-1)+horizontal_tiles*i)

        last_row = range((horizontal_tiles * vertical_tiles) - horizontal_tiles, horizontal_tiles * vertical_tiles)

        polygons = []
        for x1, y1, x2, y2 in dimensions:
            polygons.append([x1, y1, x1, y2, x2, y2, x2, y1])

        polygon_indices = []
        for i in range((vertical_tiles * horizontal_tiles) - 1):
            if i not in last_row and i not in last_column:
                polygon_indices.append([i, i + 1, i + horizontal_tiles, i + 1 + horizontal_tiles])

        for a, b, c, d in polygon_indices:
            dx = random.randint(-magnitude, magnitude)
            dy = random.randint(-magnitude, magnitude)

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[a]
            polygons[a] = [x1, y1,
                            x2, y2,
                            x3 + dx, y3 + dy,
                            x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[b]
            polygons[b] = [x1, y1,
                            x2 + dx, y2 + dy,
                            x3, y3,
                            x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[c]
            polygons[c] = [x1, y1,
                            x2, y2,
                            x3, y3,
                            x4 + dx, y4 + dy]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[d]
            polygons[d] = [x1 + dx, y1 + dy,
                            x2, y2,
                            x3, y3,
                            x4, y4]

        generated_mesh = []
        for i in range(len(dimensions)):
            generated_mesh.append([dimensions[i], polygons[i]])

        image = image.transform(image.size, Image.MESH, generated_mesh, resample=Image.BICUBIC)

        return image

class LabelTransforms():
    def __call__(self,label,stem):
        for idx,s in enumerate(list(stem)):
            if s in ['I','l']:
                if torch.rand(1) > 0.5:
                    num=all_str[s]
                    offset=np.random.uniform(0.5,1)
                    label[62*idx:62*idx+62][num]=offset
                    if s == 'l':
                        label[62*idx:62*idx+62][all_str['I']]=1-offset
                    elif s == 'I':
                        label[62*idx:62*idx+62][all_str['l']]=1-offset
        return label