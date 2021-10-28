"""
Lecture4 Fisher
Created on Thu Sep 30 17:41:15 2021
@author: ZihanGan
"""

import numpy as np
import os
from matplotlib import pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Fisher(object):
    def __init__(self,n):
        #分类器参数
        self.space=n
        self.w=np.zeros((n,1),float)
        self.s=0
    #fisher
    def train(self,X,Y):
        l=len(Y)
        num1=0
        num2=0
        sigma1=0
        sigma2=0
        miu1=np.zeros((1,self.space))
        miu2=np.zeros((1,self.space))
        for i in range(l):
            if Y[i]>0:
                num1+=1
                miu1+=X[i]
            else:
                num2+=1
                miu2+=X[i]
        miu1/=num1
        miu2/=num2
        #参数都作为行向量处理
        for i in range(l):
            if Y[i]>0:
                sigma1+=np.dot((X[i]-miu1).T,X[i]-miu1)
            else:
                sigma2+=np.dot((X[i]-miu2).T,X[i]-miu2)
        Sw=sigma1/num1+sigma2/num2
        self.w=np.dot(np.linalg.inv(Sw),(miu1-miu2).T)
        self.s=np.dot(miu1+miu2,self.w)[0][0]/2
        return
    #测试
    def test(self,X,Y):
        l=len(Y)
        loss=0
        for i in range(l):
            #sign针对二分类问题
            if self.classify(X[i])!=Y[i]:
            #if self.classify(X[i])!=Y[i]:
                loss+=1
        print("loss:",float(loss/l))
        return
    #分类
    def classify(self,x):
        return np.sign(np.dot(x,self.w))
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
                color[i][j]=(3 if (self.classify([x[i][j],y[i][j]]))==1 else 1)
        plt.contourf(x,y,color,0,alpha=0.2,colors = ['green','blue'])
        #最佳投影
        plt.quiver(3,3,self.w[0],-self.w[1], scale_units='xy', scale=1)
        plt.show()
        return

#打乱数据
def shuf(data,label):
    d=np.hstack([data,label])
    np.random.shuffle(d)
    return d[:,:-1],d[:,-1]

#产生数据
num=200
prop=0.8
x1 = np.random.multivariate_normal([5,0],[[1,0],[0,1]],num)
x2 = np.random.multivariate_normal([0,5],[[1,0],[0,1]],num)
x=np.r_[x1,x2]
#plt.scatter(x[:,0],x[:,1])
y=np.r_[np.ones((int(num),1)),-np.ones((int(num),1))]
#增广
#x=np.c_[x,np.ones((2*num,1))]
#打乱
x,y=shuf(x,y)
#割分
x_data,x_test=x[:int(2*num*prop)],x[int(2*num*prop):]
y_data,y_test=y[:int(2*num*prop)],y[int(2*num*prop):]
#分类器
model=Fisher(x_data[0].size)
#fisher
model.train(x_data,y_data)
#测试
model.test(x_test,y_test)
#model.test(x_data,y_data)
#画图
model.render(x,y)