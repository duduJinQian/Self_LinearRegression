import numpy as np
from matplotlib import pyplot as plt

class Linear():
    def __init__(self,xtrain,ytrain,xval,yval,winit,learningrate=0.001,nstep=1000):
        self.xtrain=xtrain
        self.ytrain=ytrain
        
        self.xval=xval
        self.yval=yval        
        
        self.ndim=self.xtrain.shape[1]
        ## 初始化
        self.winit=winit
        ## 可选传的
        self.learningrate=learningrate
        self.nstep=nstep
        self.initloss=0

    def linear_fun(self,x,wall):
        ypre=np.dot(x,wall.transpose())
        return ypre   

    def get_loss(self,x,wall,yreal):
        ypre=self.linear_fun(x,wall)
        gapy=(ypre-yreal)
        gloss=np.dot(gapy.transpose(),gapy)/x.shape[0]
        return gloss

    def get_gradient(self,x,wall,yreal):
        ypre=self.linear_fun(x,wall)
        gapy=(ypre-yreal)
        wgrad=np.dot(gapy.transpose(),x)/x.shape[0]
        return wgrad       

    def trans_gradient(self,x,wall,yreal,learnrate=0.01):
        wgrad=self.get_gradient(x,wall,yreal)
        w=wall-learnrate*wgrad
        return w
    
    def __call__(self):
        x=self.xtrain
        yreal=self.ytrain
        w=self.winit
        learningrate=self.learningrate
        nstep=self.nstep
        
        wnew=[]
        wnew.append(self.trans_gradient(x,w,yreal,learnrate=learningrate))
        self.initloss=self.get_loss(x,w,yreal)
        loss=[]
        loss.append(self.initloss)
        loss_val=[]
        loss_val.append(self.get_loss(self.xval,w,self.yval)[0,0])
           
        for i,n in enumerate(range(nstep)):
            wnew.append(self.trans_gradient(x,wnew[i-1],yreal,learnrate=learningrate))
            loss.append(self.get_loss(x,wnew[i-1],yreal)) 
            loss_val.append(self.get_loss(self.xval,wnew[i-1],self.yval)[0,0])
        
        plt.figure()
        plt.plot(np.array(loss_val),label='validation')
        plt.plot(np.array(loss)[:,0,0],label='train')
        plt.legend()
        plt.show()
        ### 画出val的曲线
        return wnew[-1],loss[-1]