"""
Lecture3 linear regression
Created on Thu Sep 30 17:41:15 2021
@author: ZihanGan
"""

import numpy as np
import random
import os
from matplotlib import pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class LR(object):
    def __init__(self,n):
        #分类器参数
        self.space=n
        self.w=np.zeros((n,1),float)
        #超参数
        self.lr=np.ones((n,1),float)*0.4
        self.batchsize=1
        self.epoch=1
        #开关
        self.if_ada=False
        self.if_rms=False
        self.if_momentum=True
        self.if_adam=False
        self.if_stochastic=True
        #lr方法
        self.time=0
        self.ada=np.zeros((n,1),float)
        self.epsilon=1e-6
        self.rmsprop=np.zeros((n,1),float)
        self.alpha=0.9
        self.beta1=0.99
        self.beta1_t=1
        self.v=np.zeros((n,1),float)
        #动量方法
        self.mom=np.zeros((n,1),float)
        self.Lambda=0.9
        self.beta2=0.999
        self.beta2_t=1
        #损失
        self.loss=[]
    #计算L2损失
    def Loss2(self,X,Y):
        l=len(Y)
        ls=0
        for i in range(l):
            ls+=(self.classify(X[i])-Y[i])**2
        return ls/l
    #损失曲线
    def L2_render(self):
        plt.plot(model.loss)
        plt.xlabel('epoch',fontsize=14)
        plt.ylabel('loss',fontsize=14)
        plt.title('L2',fontsize=24)
        return
    #广义逆
    def gi(self,x,y):
        xgi=np.dot(np.linalg.inv(np.dot(x.T,x)),x.T)
        return np.dot(xgi,y)
    #学习率变化
    def lr_update(self,g):
        self.time+=1
        if self.if_rms:
            for i in range(self.space):
                if self.rmsprop[i]==0:
                    self.rmsprop[i]=g[i]
                else:
                    self.rmsprop[i]=self.epsilon+np.square(self.rmsprop[i]**2*self.alpha+(1-self.alpha)*g[i]**2)
                self.lr[i]=self.lr[i]/self.rmsprop
        elif self.if_ada:
            for i in range(self.space):
                self.ada[i]+=g[i]**2
            self.lr=self.lr/(np.square(self.ada/(self.time+1))+self.epsilon)
        elif self.if_adam:
            self.v=self.v*self.beta2+(1-self.beta2)*g**2
            self.beta2_t*=self.beta2
            self.v=self.v/(1-self.beta2_t)
            self.lr=self.lr/(self.epsilon+np.square(self.v))
        return
    #梯度下降
    def train(self,X,Y):
        for i in range(self.epoch):
            if self.if_stochastic:
                for j in range(self.batchsize):
                    sel=0
                    sel=random.randint(1,len(Y)-1)
                    g=(self.classify(X[sel])-Y[sel])*X[sel]
                    self.lr_update(g)
                    if self.if_momentum:
                        self.mom=self.Lambda*self.mom-self.lr*g
                        self.w+=self.mom
                    elif self.if_adam:
                        self.mom=self.beta1*self.mom-g*(1-self.beta1)
                        self.beta1_t*=self.beta1
                        self.mom=self.mom/(1-self.beta1_t)
                        self.w-=self.mom*self.lr
                    else:
                        self.w-=g*self.lr
            else:
                u=0
                ep=0
                #numpy数组*点乘一直报错，非常奇怪，好像视为矩阵乘法了
                for i in range(len(Y)):
                    if ep==self.batchsize:
                        u/=self.batchsize*2
                        u=np.matrix(u).T
                        self.lr_update(u)
                        if self.if_momentum:
                            self.mom=self.Lambda*self.mom-self.lr*u
                            self.w+=self.mom
                        elif self.if_adam:
                            self.mom=self.beta1*self.mom-u*(1-self.beta1)
                            self.beta1_t*=self.beta1
                            self.mom=self.mom/(1-self.beta1_t)
                            self.w-=self.mom*self.lr
                        else:
                            #把*换为multiply
                            self.w-=np.multiply(u,self.lr)
                        u=(self.classify(X[i])-Y[i])*X[i]
                        ep=1
                    else:
                        u+=(self.classify(X[i])-Y[i])*X[i]
                        ep+=1
            self.loss.append(self.Loss2(X,Y))
        return
    #测试
    def test(self,X,Y):
        l=len(Y)
        loss=0
        for i in range(l):
            #sign针对二分类问题
            if np.sign(self.classify(X[i]))!=Y[i]:
            #if self.classify(X[i])!=Y[i]:
                loss+=1
        print("loss:",float(loss/l))
        return
    #分类
    def classify(self,x):
        return np.dot(x,self.w)
    #画图，仅对于二维特征
    def render(self,X,Y):
        #取范围
        x_min1,x_max1=np.amin(X[:,0]),np.amax(X[:,0])
        x_min2,x_max2=np.amin(X[:,1]),np.amax(X[:,1])
        #散点
        plt.xlabel('feature1',fontsize=14)
        plt.ylabel('feature2',fontsize=14)
        plt.title('data',fontsize=24)
        #蓝色类别1
        x=[X[i][0] for i in range(len(X)) if Y[i]==1]
        y=[X[i][1] for i in range(len(X)) if Y[i]==1]
        plt.scatter(x, y, c='blue',edgecolor='grey',s=200, marker='$\clubsuit$', alpha=0.8)
        #绿色类别2
        x=[X[i][0] for i in range(len(X)) if Y[i]==-1]
        y=[X[i][1] for i in range(len(X)) if Y[i]==-1]
        plt.scatter(x, y, c='green',edgecolor='red', s=200,marker='$\heartsuit$', alpha=0.8)
        #分类面
        rec=100
        x,y=np.meshgrid(np.linspace(x_min1,x_max1, rec),np.linspace(x_min2,x_max2, rec))
        color=np.zeros((rec,rec))
        for i in range(rec):
            for j in range(rec):
                color[i][j]=(3 if np.sign((self.classify([x[i][j],y[i][j],1])))==1 else 1)
        plt.contourf(x,y,color,0,alpha=0.2,colors = ['green','blue'])
        plt.show()
        return

#打乱数据
def shuf(data,label):
    d=np.hstack([data,label])
    np.random.shuffle(d)
    return d[:,:-1],d[:,-1]

#自定义单变量函数
def fun(x,isgd=False):
    if isgd:
        return np.cos(0.25*np.pi*x)-0.25*np.pi*x*np.sin(0.25*np.pi*x)
    else:
        return x*np.cos(0.25*np.pi*x)

#折线图
def lc(x,y):
    plt.xlabel('x',fontsize=14)
    plt.ylabel('f(x)',fontsize=14)
    plt.title('adam',fontsize=24)
    plt.quiver(x[:-1], y[:-1], x[1:]-x[:-1], y[1:]-y[:-1], scale_units='xy', angles='xy', scale=1)
    plt.show()
    return

#产生数据
num=200
prop=0.8
x1 = np.random.multivariate_normal([2,0],[[1,0],[0,1]],num)
x2 = np.random.multivariate_normal([0,2],[[1,0],[0,1]],num)
x=np.r_[x1,x2]
#plt.scatter(x[:,0],x[:,1])
y=np.r_[np.ones((int(num),1)),-np.ones((int(num),1))]
#增广
x=np.c_[x,np.ones((2*num,1))]
#打乱
x,y=shuf(x,y)
#割分
x_data,x_test=x[:int(2*num*prop)],x[int(2*num*prop):]
y_data,y_test=y[:int(2*num*prop)],y[int(2*num*prop):]
#分类器
model=LR(len(x_data[0]))
#广义逆
#model.w=model.gi(x_data, y_data)
#梯度下降
model.train(x_data,y_data)
#测试
model.test(x_test,y_test)
#model.test(x_data,y_data)
#画图
model.render(x_test,y_test)
#model.L2_render()
#model.render(x_data,y_data)
'''
model=LR(1)
model.w=[[-4]]
t=50
wgh=np.zeros((t,1))
y=np.zeros((t,1))
for i in range(t):
    model.train(model.w[0],[0])
    wgh[i]=model.w[0][0]
    y[i]=fun(wgh[i])
lc(wgh,y)
'''