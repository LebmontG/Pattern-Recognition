"""
Lecture9 Classification for Multiclass
Created on Sun Oct 17 10:38:37 2021
@author: ZihanGan
"""

# -*- coding: utf-8 -*-
import struct
import os
from PLA import PLA
import numpy as np
from matplotlib import pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#从Lecture9开始可以使用机器学习库(解放日)
import torch.nn as nn
import torch
import torch.optim as optim

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

#打乱数据
def shuf(data,label):
    l=len(label)
    ind=[i for i in range(l)]
    np.random.shuffle(ind)
    return data[ind],label[ind]

class OVO(object):
    def __init__(self,n,dim):
        #类别数量
        self.class_num=n
        self.class_name={i:chr(i+1) for i in range(n)}
        #记录分类器所分的两个类别
        self.model_loc=dict()
        j=0
        k=1
        for i in range(n*(n-1)//2):
            self.model_loc[i]=[j,k]
            if k==n-1:
                j+=1
                k=j+1
            else:
                k+=1
        #PLA分类器
        self.model_num=n*(n-1)//2
        self.model=[PLA(dim) for _ in range(n*(n-1)//2)]
    def classify(self,x):
        vote=np.zeros((self.class_num,1))
        for i in range(self.class_num):
            if self.model[i].discern(x)==1:
                vote[self.model_loc[i][0]]+=1
            else:
                vote[self.model_loc[i][1]]+=1
        return self.class_name[np.argmax(vote)]
    def train(self,X,Y,ptn):
        l=len(Y)
        for i in range(self.class_num):
            self.class_name[i]=ptn[i]
        XX=[[] for i in range(self.class_num)]
        test=[[] for i in range(self.class_num)]
        for i in range(l):
            for j in range(self.class_num):
                if Y[i]==self.class_name[j]:
                    XX[j].append(X[i])
                    break
        for i in range(self.class_num):
            XX[i],_=shuf(np.array(XX[i]),np.zeros((len(XX[i]),1)))
            test[i],XX[i]=XX[i][30:],XX[i][0:30]
        for i in range(self.model_num):
            x=np.r_[XX[self.model_loc[i][0]],XX[self.model_loc[i][1]]]
            y=np.r_[np.ones((len(XX[self.model_loc[i][0]]),1)),-np.ones((len(XX[self.model_loc[i][1]]),1))]
            self.model[i].train(x,y)
        #测试
        l=0
        for i in range(self.class_num):
            for ele in test[i]:
                if self.classify(ele)==self.class_name[i]:
                    l+=1
        print("accuracy:",1-l/len(test)/test[0].size)
        return

class softmax(object):
    def __init__(self,input_dim,output_dim):
        self.input_dim =input_dim
        self.output_dim = output_dim
        self.device='cpu'
        self.epoches=10
        self.batchsize=256
        self.fc1 = nn.Linear(input_dim,output_dim)
        self.optimizer=optim.SGD(self.fc1.parameters(), lr=0.3)
        #self.optimizer.param_groups
        self.loss=nn.CrossEntropyLoss()
        self.losses=[]
        self.acc_test=[]
        self.acc_train=[]
        return
    def train(self,x,y,x_t,y_t):
        for i in range(self.epoches):
            cel=0
            loc=0
            while(loc+self.batchsize<len(y)):
                self.optimizer.zero_grad()
                tar=self.forward(x[loc:loc+self.batchsize])
                l=self.loss(tar,y[loc:self.batchsize+loc].T).sum()/self.batchsize
                l.backward()
                cel+=l
                self.optimizer.step()
                loc+=self.batchsize
            print("episodes:%d\\loss:%f"%(i,cel))
            self.losses.append(cel.__float__())
            self.acc_train.append(self.test(x,y))
            self.acc_test.append(self.test(x_t,y_t))
        return
    def forward(self,x):
        return nn.functional.softmax(self.fc1(x),dim=1)
    def test(self,x,y):
        l=len(y)
        res=[torch.argmax(ele) for ele in self.forward(x)]
        return sum((np.array(res)==y.numpy()))/l
    def L_render(self,x,y,ob):
        plt.plot(ob)
        plt.xlabel(x,fontsize=14)
        plt.ylabel(y,fontsize=14)
        plt.show()
        return

def shuf(data,label):
    d=np.hstack([data,label])
    np.random.shuffle(d)
    return d[:,:-1],d[:,-1]

'''
f=pd.read_csv('Iris\\iris.csv')
ptn=list(set(f.iloc[:,-1]))
l=len(ptn)
X=np.array(f.iloc[:,1:-1])
Y=np.array(f.iloc[:,-1])
m=OVO(len(ptn),X[0].size)
m.train(X, Y, ptn)
#随机选30个数据训练
x_train=[[] for i in range(l)]
x_test=[[] for i in range(l)]
for i in range(len(Y)):
    for j in range(l):
        if Y[i]==ptn[j]:
            x_train[j].append(X[i])
for i in range(l):
    x_train[i],_=shuf(np.array(x_train[i]),np.zeros((len(x_train[i]),1)))
    x_test[i],x_train[i]=x_train[i][30:],x_train[i][0:30]
m=softmax(len(x_train[i][0]),len(ptn))
#nn.init.normal_(m.fc1.weight,mean=0,std=0.01)
for _ in range(m.epoches):
    for i in range(l):
        y=torch.LongTensor(np.ones((len(x_train[i]),1))*i)
        y=torch.autograd.Variable(y)
        x=torch.autograd.Variable(torch.FloatTensor(x_train[i]))
        m.train(x,y)
acu=0
ac=0
for i in range(l):
    y=torch.LongTensor(np.ones((len(x_test[i]),1))*i)
    y=torch.autograd.Variable(y)
    x=torch.autograd.Variable(torch.FloatTensor(x_test[i]))
    acu+=m.test(x,y)*len(x_test[i])
    ac+=len(x_test[i])
print("accuracy:",acu/ac)
#print(m.optimizer.param_groups)
'''
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
m=softmax(len(img[0]),10)
nn.init.normal_(m.fc1.weight,mean=0,std=0.01)
m.train(torch.Tensor(img),torch.LongTensor(label),torch.Tensor(timg),torch.LongTensor(tlabel))
print(m.test(torch.Tensor(timg[:10]),torch.LongTensor(tlabel[:10])))
m.L_render('epoch','acc_train',m.acc_train)
m.L_render('epoch','acc_test',m.acc_test)
m.L_render('epoch','losses',m.losses)
