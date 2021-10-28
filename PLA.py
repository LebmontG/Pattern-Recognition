"""
Lecture2 PLA/Pocket
Created on Tue Sep 14 12:33:38 2021
@author: ZihanGan
"""

import numpy as np
import random
import time

class PLA(object):
    def __init__(self,n):
        #超参数
        self.lr=1
        self.batch=12
        self.PLA_epi=500
        self.pocket_epi=500
        #分类器参数
        self.space=n
        self.w=np.zeros((n,1),float)
    #真-分类
    def discern(self,x):
        return np.sign(np.dot(x,self.w))
    #分类
    def recog(self,x,y,w):
        xx=np.matrix(x)
        y_pre=np.sign(np.dot(xx,w))
        return y_pre==y
    #更新
    def fit(self,w,batch):
        x,y=batch[0][:],batch[1]
        for i in range(len(x)):
            w[i]+=x[i]*self.lr*y
        return w
    #PLA
    def train(self,X,Y):
        time_start=time.time()
        l=len(Y)
        loss=1
        ep=0
        while(loss>0):
            loss=0
            for i in range(l):
                if self.recog(X[i],Y[i],self.w)==0:
                    loss+=1
                    self.w=self.fit(self.w,(X[i],Y[i]))
            #防止无限
            if ep<self.PLA_epi:
                ep+=1
            else:
                loss=sum([int(self.recog(X[i],Y[i],self.w)==0) for i in range(l)])
                break
        #print("pla_loss:",float(loss/l))
        #self.render(X,Y)
        loss=0
        time_end=time.time()
        #print('PLA time',time_end-time_start)
        return
    #pockets
    def pocket(self,X,Y):
        time_start=time.time()
        l=len(Y)
        loss_pre=l
        er=[i for i in range(l)]
        for _ in range(self.pocket_epi):
            upd=random.sample(er,1)[0]
            #print(self.w,loss_pre,X[upd])
            loss=0
            er_n=[]
            w=self.fit(self.w,(X[upd],Y[upd]))
            for i in range(l):
                if self.recog(X[i],Y[i],w)==0:
                    loss+=1
                    er_n.append(i)
            if loss==0:
                break
            if loss<loss_pre:
                loss_pre=loss
                self.w=w[:]
                er=er_n[:]
        print("pockets_loss:",loss/l)
        time_end=time.time()
        print('pockets time',time_end-time_start)
        #self.render(X,Y)
        return
    def classify(self,X,Y):
        l=len(Y)
        loss=0
        for i in range(l):
            if self.recog(X[i],Y[i],self.w)==0:
                loss+=1
        print("loss:",float(loss/l))
        return