"""
Lecture3 support vector machine
Created on Sat Oct  9 19:15:24 2021
@author: ZihanGan
"""

import numpy as np
import cvxopt
import os
from matplotlib import pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class SVM(object):
    def __init__(self,d):
        #svm参数
        self.n=d
        self.w=np.zeros((d,1),float)
        #超参数
        self.ksai=10
        self.e=4
        self.gamma=0.01
        #kernel分类
        self.ker=[]
        self.kernel_func='gauss'
    def classify(self,x,kernel=None):
        if kernel==None:
            return np.dot(x,self.w[:-1])+self.w[-1]
    def test(self,X,Y,K=0):
        l=len(Y)
        loss=0
        for i in range(l):
            #sign针对二分类问题
            if K:
                if self.k_cls(X[i])!=Y[i]:
                    loss+=1
            else:
                if np.sign(self.classify(X[i]))!=Y[i]:
                    #if self.classify(X[i])!=Y[i]:
                    loss+=1
        print("loss:",float(loss/l))
        return
    def k_cls(self,x):
        return np.sign(sum([self.ker[i][0]*self.K_func(self.ker[i][1:][0],x) for i in range(len(self.ker))])+self.w[-1])
    #变换核
    def K_func(self,X,Y):
        if self.kernel_func=='gauss':
            v=X-Y
            return np.exp(-np.dot(v,v.T))
        else:
            #假设是两个行向量
            return (self.ksai+self.gamma*np.dot(X,np.array(Y).T))**self.e
    #利用二次规划而非gd的方法实现三种svm
    def dual(self,X,Y):
        l=len(Y)
        p=-1.0*np.ones((l,1))
        Q=np.zeros((l,l))
        for i in range(l):
            for j in range(l):
                Q[i][j]=Y[i]*Y[j]*np.dot(X[i],X[j].T)
        A=1.0*Y.T
        b=0.0
        G=-1.0*np.eye(l)
        h=1.0*np.zeros((l,1))
        #A=[np.multiply(Y,X[:,i]) for i in range(self.n)]
        #必须用cvxopt自带的matrix类型，注意数据类型要相符
        Q=cvxopt.matrix(Q)
        p=cvxopt.matrix(p)
        G=cvxopt.matrix(G)
        h=cvxopt.matrix(h)
        A=cvxopt.matrix(A)
        b=cvxopt.matrix(b)
        #res是多个结果的字典
        res=cvxopt.solvers.qp(Q, p, G,h,A.T,b)
        alpha=res['x']
        #得到支持向量之后用X求w和b
        for i in range(l):
            if alpha[i]>1e-5:
                self.w[:-1]+=alpha[i]*np.matrix(np.multiply(Y[i],X[i])).T
        for i in range(l):
            if alpha[i]>1e-5:
                self.w[-1]=Y[i]-np.dot(X[i],self.w[:-1])
                break
        return res['x']
    def primal(self,X,Y):
        #增广
        X=np.c_[X,np.ones((X.size//X[0].size,1))]
        p=np.zeros((self.n,1))
        Q=np.zeros((self.n,self.n))
        for i in range(self.n-1):
            Q[i][i]=1
        c=-1*np.ones((len(Y),1))
        #直接相乘，实际上是X按列与Y相乘
        #这里*一直报错，只能手动乘
        A=(-1.0*np.multiply(np.matrix(Y).T,X))
        #A=[np.multiply(Y,X[:,i]) for i in range(self.n)]
        #必须用cvxopt自带的matrix类型，注意数据类型要相符
        Q=cvxopt.matrix(Q)
        p=cvxopt.matrix(p)
        A=cvxopt.matrix(A)
        c=cvxopt.matrix(c)
        #res是多个结果的字典
        res=cvxopt.solvers.qp(Q, p, G=A,h=c)
        #X的增广是在尾部添加1，因此结果的x=[w,b]
        self.w=np.matrix(res['x'])
        return
    def kernel(self,X,Y):
        l=len(Y)
        p=-1.0*np.ones((l,1))
        Q=np.zeros((l,l))
        for i in range(l):
            for j in range(l):
                Q[i][j]=Y[i]*Y[j]*self.K_func(X[i],X[j])
        A=1.0*Y.T
        b=0.0
        G=-1.0*np.eye(l)
        h=1.0*np.zeros((l,1))
        #A=[np.multiply(Y,X[:,i]) for i in range(self.n)]
        #必须用cvxopt自带的matrix类型，注意数据类型要相符
        Q=cvxopt.matrix(Q)
        p=cvxopt.matrix(p)
        G=cvxopt.matrix(G)
        h=cvxopt.matrix(h)
        A=cvxopt.matrix(A)
        b=cvxopt.matrix(b)
        #res是多个结果的字典
        res=cvxopt.solvers.qp(Q, p, G,h,A,b)
        alpha=res['x']
        #得到支持向量之后用X求w和b
        sig=0
        for i in range(l):
            if alpha[i]>1e-5:
                self.ker.append([alpha[i]*Y[i],X[i]])
                sig+=alpha[i]*Y[i]*self.K_func(X[i],X[0])
                #self.w[:-1]+=alpha[i]*np.matrix(np.multiply(Y[i],X[i])).T
        self.w[-1]=Y[0]-sig
        self.ker=np.array(self.ker)
        return res['x']
    def render(self,X,Y,alpha=None):
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
        #plt.scatter(124,25.4, c='black',edgecolor='red', s=300,marker='o', alpha=0.8)
        #找出sv，设定一个阈值
        # if alpha!=None:
        #     x=np.array([x_data[i] for i in range(y_data.size//2) if alpha[i]>1e-5])
        #     y=np.array([y_data[i] for i in range(y_data.size//2) if alpha[i]>1e-5])
        #     plt.scatter(x[:,0], x[:,1], c='yellow', s=150,marker='o')
        #分类面
        rec=100
        x,y=np.meshgrid(np.linspace(x_min1,x_max1, rec),np.linspace(x_min2,x_max2, rec))
        color=np.zeros((rec,rec))
        for i in range(rec):
            for j in range(rec):
                color[i][j]=(3 if self.k_cls([x[i][j],y[i][j]])==1 else 1)
                #间隔面
                if abs(self.k_cls([x[i][j],y[i][j]]))<1:
                    color[i][j]=5
        plt.contourf(x,y,color,1,alpha=0.2,colors = ['red','blue','yellow'])
        plt.show()
        return

#打乱数据
def shuf(data,label):
    d=np.hstack([data,label])
    np.random.shuffle(d)
    return d[:,:-1],d[:,-1]
'''
x_data=np.matrix([[2,2],[-2,-2],[2,-2],[-2,2]])
y_data=np.matrix([[1],[1],[-1],[-1]])
'''
#产生数据
num=200
prop=0.8
x1 = np.random.multivariate_normal([3,0],[[1,0],[0,1]],num)
x2 = np.random.multivariate_normal([0,3],[[1,0],[0,1]],num)
x=np.r_[x1,x2]
y=np.r_[np.ones((int(num),1)),-np.ones((int(num),1))]
#打乱
x,y=shuf(x,y)
#割分
x_data,x_test=x[:int(2*num*prop)],x[int(2*num*prop):]
y_data,y_test=y[:int(2*num*prop)],y[int(2*num*prop):]
#分类器
model=SVM(x_data[0].size+1)
# #svm
# model.primal(x_data,y_data)
# alpha=[0 for i in range(y_data.size)]
# for i in range(y_data.size):
#     if abs(abs(model.classify([x_data[i][0],x_data[i][1]]))-1)<0:
#         alpha[i]=1
#     else:
#         alpha[i]=0
#alpha=model.dual(x_data,y_data)
alpha=model.kernel(x_data,y_data)
#测试
#model.test(x_test,y_test)
#model.test(x_data,y_data)
model.test(x_test,y_test,1)
#model.test(x_data,y_data,1)
#画图
model.render(x_data,y_data,alpha)
#model.render(x,y)
#model.render(x_data,y_data)
'''
x1=np.array([[119.28,26.08],#福州
     [121.31,25.03],#台北
     [121.47,31.23],#上海
     [118.06,24.27],#厦门
     [121.46,39.04],#大连
     [122.10,37.50],#威海
     [124.23,40.07]])#丹东
x2=np.array([[129.87,32.75],#长崎
     [130.33,31.36],#鹿儿岛
     [131.42,31.91],#宫崎
     [130.24,33.35],#福冈
     [133.33,15.43],#鸟取
     [138.38,34.98],#静冈
     [140.47,36.37]])#水户
x_data=np.r_[x1,x2]
y_data=np.r_[np.ones((10,1)),-np.ones((10,1))]
test=[124,25.4]
x1=np.array([[119.28,26.08],#福州
         [121.31,25.03],#台北
         [121.47,31.23],#上海
         [118.06,24.27],#厦门
         [113.53,29.58],#武汉
         [104.06,30.67],#成都
         [116.25,39.54],#北京
         [121.46,39.04],#大连
         [122.10,37.50],#威海
         [124.23,40.07]])#丹东
x2=np.array([[129.87,32.75],#长崎
         [130.33,31.36],#鹿儿岛
         [131.42,31.91],#宫崎
         [130.24,33.35],#福冈
         [136.54,35.10],#名古屋
         [132.27,34.24],#广岛
         [139.46,35.42],#东京
         [133.33,15.43],#鸟取
         [138.38,34.98],#静冈
         [140.47,36.37]])#水户
'''