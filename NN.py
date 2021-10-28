"""
Lecture10-11 Neural Network
Created on Sun Oct 24 20:52:06 2021
@author: ZihanGan
"""

# -*- coding: utf-8 -*-
import pandas as pd
import struct
import os
import numpy as np
from matplotlib import pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.lr=0.1
        self.epoches=10
        self.batchsize=256
        self.fc1 = nn.Conv2d(1,6,5,stride=1, padding=2)
        self.fc2 = nn.AvgPool2d((2,2),stride=[2])
        self.fc3 = nn.Conv2d(6,16,5,stride=[1], padding=[0])
        self.fc4 = nn.AvgPool2d((2,2),stride=[2])
        self.fc5 = nn.Conv2d(16,120,5)
        self.flat=nn.Flatten()
        self.fc6 = nn.Linear(120,84)
        self.fc7 = nn.Linear(84,10)
        self.optimizer=optim.SGD(self.parameters(),self.lr)
        self.loss=nn.CrossEntropyLoss()
        self.losses=[]
        self.ac_test=[]
        self.ac_train=[]
        #self.optimizer.param_groups
    def render(self,l,name):
        plt.plot(l)
        plt.xlabel('epoch',fontsize=14)
        plt.ylabel('loss',fontsize=14)
        plt.title(name,fontsize=24)
        plt.show()
        return
    def train(self,x,y,xt,yt):
        for i in range(self.epoches):
            loc=0
            L=0
            while(loc+self.batchsize<len(y)):
                inp=torch.zeros(256,1,28,28)
                for j in range(self.batchsize):
                    inp[j][0]=x[loc+j].reshape(28,28)
                self.optimizer.zero_grad()
                l=self.loss(self.forward(inp),y[loc:loc+self.batchsize])
                l.backward()
                self.optimizer.step()
                loc+=self.batchsize
                L+=float(l)
            self.losses.append(L)
            self.ac_test.append(self.test(xt,yt))
            self.ac_train.append(self.test(x,y))
            print(i+1,self.losses[-1])
        return
    def test(self,x,y):
        a=0
        l=len(y)
        for i in range(l):
            inp=torch.zeros(1,1,28,28)
            inp[0][0]=x[i].reshape(28,28)
            a+=y[i].item()==torch.argmax(self.forward(inp))
        return a/l
    def forward(self, x):
        # 各层对应的激活函数
        x = F.sigmoid(self.fc1(x))
        x =self.fc2(x)
        x = F.sigmoid(self.fc3(x))
        x =self.fc4(x)
        x=self.fc5(x)
        x=self.flat(x)
        x = F.sigmoid(self.fc6(x))
        return F.softmax(self.fc7(x),dim=1)

class NeuralNet(nn.Module):
    def __init__(self, input_dim,output_dim,hidden_dim=32):
        super(NeuralNet, self).__init__()
        self.lr=0.01
        self.epoches=100
        self.batchsize=4
        self.fc1 = nn.Linear(input_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        self.fc3 = nn.Linear(hidden_dim,hidden_dim)
        self.fc4 = nn.Linear(hidden_dim,hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, output_dim)
        self.optimizer=optim.SGD(self.parameters(),self.lr)
        self.loss=nn.CrossEntropyLoss()
        self.losses=[]
        #self.optimizer.param_groups
    def render(self):
        plt.plot(self.losses)
        plt.xlabel('epoch',fontsize=14)
        plt.ylabel('loss',fontsize=14)
        plt.title('LCE',fontsize=24)
        plt.show()
        return
    def train(self,x,y):
        for i in range(self.epoches):
            loc=0
            L=0
            while(loc+self.batchsize<len(y)):
                self.optimizer.zero_grad()
                l=self.loss(self.forward(x[loc:loc+self.batchsize]),y[loc:loc+self.batchsize])
                l.backward()
                self.optimizer.step()
                loc+=self.batchsize
                L+=float(l)
            self.losses.append(L)
        return
    def forward(self, x):
        # 各层对应的激活函数
        x = F.sigmoid(self.fc1(x)) 
        x = F.sigmoid(self.fc2(x)) 
        x = F.sigmoid(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        return F.softmax(self.fc5(x))

class MNIST(object):
    def __init__(self,root, image_file, lable_file):
        '''
            root: 文件夹根目录
            image_file: mnist图像文件 'train-images.idx3-ubyte' 'test-images.idx3-ubyte'
            label_file: mnist标签文件 'train-labels.idx1-ubyte' 'test-labels.idx1-ubyte'
        '''
        self.img_file = os.path.join(root, image_file)
        self.label_file = os.path.join(root, lable_file)
        self.img = self._get_img()
        self.label = self._get_label()
    #读取图片
    def _get_img(self):
        with open(self.img_file,'rb') as fi:
            ImgFile = fi.read()
            head = struct.unpack_from('>IIII', ImgFile, 0)
            #定位数据开始位置
            offset = struct.calcsize('>IIII')
            ImgNum = head[1]
            width = head[2]
            height = head[3]
            #每张图片包含的像素点
            pixel = height*width
            bits = ImgNum * width * height
            bitsString = '>' + str(bits) + 'B'
            #读取文件信息
            images = struct.unpack_from(bitsString, ImgFile, offset)
            #转化为n*726矩阵
            images = np.reshape(images,[ImgNum,pixel])
        return images
    #读取标签
    def _get_label(self):
        with open(self.label_file,'rb') as fl:
            LableFile = fl.read()
            head = struct.unpack_from('>II', LableFile, 0)
            labelNum = head[1]
            #定位标签开始位置
            offset = struct.calcsize('>II')
            numString = '>' + str(labelNum) + "B"
            labels = struct.unpack_from(numString, LableFile, offset)
            #转化为1*n矩阵
            labels = np.reshape(labels, [labelNum])
        return labels
    #数据标准化
    def normalize(self):
        min = np.min(self.img, axis=1).reshape(-1,1)
        max = np.max(self.img, axis=1).reshape(-1,1)
        self.img = (self.img - min)/(max - min)
    #数据归一化
    def standardlize(self):
        mean = np.mean(self.img, axis=1).reshape(-1,1)
        var = np.var(self.img, axis=1).reshape(-1,1)
        self.img = (self.img-mean)/np.math.sqrt(var)

def iris(path):
    f=pd.read_csv(path)
    ptn=list(set(f.iloc[:,-1]))
    l=len(ptn)
    X=np.array(f.iloc[:,1:-1])
    Y=np.array(f.iloc[:,-1])
    x_train=[[] for i in range(l)]
    #分出三类
    for i in range(len(Y)):
        for j in range(l):
            if Y[i]==ptn[j]:
                x_train[j].append(X[i])
    X=[]
    Y=[]
    X_t=[]
    Y_t=[]
    for i in range(l):
        x_train[i],_=shuf(np.array(x_train[i]),np.zeros((len(x_train[i]),1)))
        X_t[0:0]=x_train[i][30:]
        X[0:0]=x_train[i][0:30]
        Y_t+=[i for _ in range(len(x_train[i][30:]))]
        Y+=[i for _ in range(30)]
    return torch.FloatTensor(X),torch.LongTensor(Y),torch.FloatTensor(X_t),torch.LongTensor(Y_t)

#打乱数据
def shuf(data,label):
    l=len(label)
    ind=[i for i in range(l)]
    np.random.shuffle(ind)
    return data[ind],label[ind]

x,y,x_t,y_t=iris('Iris\\iris.csv')
model=NeuralNet(4,10)
model.train(x,y)
model.render()
#读入数据
dataset = MNIST(os.getcwd(),'mnist\\train-images-idx3-ubyte','mnist\\train-labels-idx1-ubyte')
dataset.normalize()
img = dataset.img
label = dataset.label
td = MNIST(os.getcwd(),'mnist\\t10k-images-idx3-ubyte','mnist\\t10k-labels-idx1-ubyte')
td.normalize()
timg = td.img
tlabel =td.label
timg,tlabel=shuf(timg,tlabel.reshape(tlabel.size,1))
m=LeNet()
m.train(torch.Tensor(img),torch.LongTensor(label),torch.Tensor(timg),torch.LongTensor(tlabel))
m.render(m.losses,'LCE')
m.render(m.ac_test,'ac_test')
m.render(m.ac_train,'ac_train')
for i in range(10):
    loc=np.random.random_integers(0,10000)
    inp=torch.zeros(1,1,28,28)
    inp[0][0]=torch.Tensor(timg[loc]).reshape(28,28)
    print("i+1:","test",tlabel[loc],"predict",torch.argmax(m.forward(inp)))