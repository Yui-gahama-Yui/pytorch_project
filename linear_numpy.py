import numpy as np
import matplotlib.pyplot as plt

#定义模型超参数(hyper parameters)
epochs=100
learning_rate=0.01

#已知数据集
xs=np.array([1,2,3])
ys=np.array([2,4,6])

#参数修正
#y=w*x
#l=loss_y=(pred_y-y)^2=(w*x-y)^2
#dl/dw=2*(w*x-y)*x=2*x*(wx-y)
#w=w-learning_rate*dl/dw

#lim dx->0: w=(f(x+dx)-f(x))/dx

#模型，面向对象
class Model():#括号内为继承的父对象，此处没有父对象
    def __init__(self):#构造函数
        #成员变量
        self.w=0
    def forward(self,x):#前向,计算pred_y
        return self.w*x
    def loss_fn(self,x,y):#计算loss
        return (self.w*x-y)**2
    def gradient(self,x,y):#反向,计算梯度
        return 2*x*(self.w*x-y)

model=Model()#模型实例化

#train model
for epoch in range(epochs):
    for step,batch in enumerate(zip(xs,ys)):
        batch_x,batch_y=batch
        pred_y=model.forward(batch_x)
        loss=model.loss_fn(batch_x,batch_y)
        grad=model.gradient(batch_x,batch_y)
        model.w=model.w-learning_rate*grad
        print("Epoch:{},Step:{},Loss:{:.4f}".format(epoch,step,loss))

#test model
test_x=5
pred_y=model.forward(test_x)
print(pred_y)