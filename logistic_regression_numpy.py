import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import sigmoid

from test_1_CNN import dataloader
from sklearn import metrics

num_epochs = 100
learning_rate = 0.001
batch_size = 4

def generate_data():
    x1=np.array([1,4],dtype=np.float64)
    x2=np.array([1,4],dtype=np.float64)
    y=np.array([0,1])

    x1=np.repeat(x1,100)
    x2=np.repeat(x2,100)
    y=np.repeat(y,100)

    x1 += np.random.randn(x1.shape[0])*1.0
    x2 += np.random.randn(x2.shape[0])*1.0

    return x1,x2,y

def split_data(x1,x2,y,train_rate=0.8):
    index=np.arange(len(x1.shape[0]))# x1.shape[0]==200
    np.random.shuffle(index)

    x1=x1[index]
    x2=x2[index]
    y=y[index]
    num_train=int(x1.shape[0]*train_rate)# num_train==160
    train_x1=x1[:num_train]# 前160个数据
    train_x2=x2[:num_train]
    train_y=y[:num_train]

    test_x1=x1[num_train:]# 后60个数据
    test_x2=x2[num_train:]
    test_y=y[num_train:]

    return train_x1,train_x2,train_y,test_x1,test_x2,test_y

class DataLoader():
    def __init__(self,x1,x2,y,batch_size):
        self.x1=x1
        self.x2=x2
        self.y=y
        self.batch_size=batch_size
    def get_batch(self,batch_index):
        start=batch_index*self.batch_size
        end=min(len(self.x1),(batch_index+1)*self.batch_size)
        return self.x1[start:end],self.x2[start:end],self.y[start:end]

def test_dataloader():
    x1, x2, y = generate_data()
    data = split_data(x1, x2, y, train_rate=0.8)
    train_x1, train_x2, train_y, test_x1, test_x2, test_y = data
    batch_size=4
    dataloader=DataLoader(train_x1,train_x2,train_y,batch_size)
    num_batches=len(train_x1)//batch_size
    for i in range(num_batches):
        batch=dataloader.get_batch(i)
        print(batch)

# 逻辑回归模型
class LogisticRegressionModel():
    def __init__(self):
        self.w1=0
        self.w2=0
        self.b=0
        self.grad_w1=0
        self.grad_w2=0
        self.grad_b=0
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    def forward(self,x1,x2):
        z=self.w1*x1 + self.w2*x2 + self.b
        return self.sigmoid(z)
    def loss(self,x1,x2,y):
        l=y*np.log(self.forward(x1,x2)) + (1-y)*np.log(1-self.forward(x1,x2))
        return -np.mean(l)
    def backward(self,x1,x2,y):
        self.grad_w1=np.sum((self.forward(x1,x2)-y)*x1)
        self.grad_w2=np.sum((self.forward(x1,x2)-y)*x2)
        self.grad_b=np.sum(self.forward(x1,x2)-y)
    def step(self,lr):
        self.w1=self.w1-lr*self.grad_w1
        self.w2=self.w2-lr*self.grad_w2
        self.b=self.b-lr*self.grad_b

def train(model,dataloader):
    num_batch=len(dataloader.x1)//batch_size
    for epoch in range(num_batch):
        for step in range(num_batch):
            batch=dataloader.get_batch(step)
            batch_x1,batch_x2,batch_y=batch
            pred_y=model.forward(batch_x1,batch_x2)
            loss=model.loss(batch_x1,batch_x2,batch_y)

            model.backward(batch_x1,batch_x2,batch_y)# 算梯度
            model.step(learning_rate)# 更新参数

            print("Epoch:{},Step:{},Loss:{}".format(epoch,step,loss))

def test(model,dataloader):
    pred_y=[]
    true_y=[]
    num_batch=len(dataloader.x1)//batch_size
    for step in range(num_epochs):
        batch=dataloader.get_batch(step)
        batch_x1,batch_x2,batch_y=batch
        batch_pred_y=model.forward(batch_x1,batch_x2)
        pred_y.extend(batch_pred_y.tolist())# extend()方法会把后面数组中的元素拆开，一个一个的加入

def draw_boundary(model,x):
    return -((model.w1/model.w2)*x+model.b/model.w2)

if __name__=='__main__':
    x1,x2,y=generate_data()
    data=split_data(x1,x2,y,train_rate=0.8)
    train_x1,train_x2,train_y,test_x1,test_x2,test_y = data
    train_dataloader=DataLoader(train_x1,train_x2,train_y,batch_size)
    test_dataloader=DataLoader(test_x1,test_x2,test_y,batch_size)

    model=LogisticRegressionModel()
    train(model,train_dataloader)

    plt.scatter(x1,x2,c=y)

    line_x1=[]
    line_x2=[]
    for i in range(0,6,0.1):
        j=draw_boundary(model,i)
        line_x1.append(i)
        line_x2.append(j)
        plt.plot(line_x1,line_x2)

    plt.show()
    #test_dataloader()
    # x1,x2,y=generate_data()
    # data=split_data(x1,x2,y,train_rate=0.8)
    # train_x1,train_x2,train_y,test_x1,test_x2,test_y=data
    print('finish')

# plt.scatter(x1,x2,c=y)# 画散点图
# plt.show()