import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader

max_epochs = 100
lr=1e-2

xs=np.array([1,2,3])
ys=np.array([2,4,6])

class MyDataset(Dataset):
    def __init__(self, xs, ys):
        super(MyDataset, self).__init__()
        self.x_tensor=torch.tensor(xs)
        self.y_tensor=torch.tensor(ys)
    #callback function
    def __len__(self):
        return self.x_tensor.shape[0]
    def __getitem__(self, index):
        return self.x_tensor[index], self.y_tensor[index]

#test MyDataset
# dataset=MyDataset(xs,ys)
# dataloader = DataLoader(dataset,batch_size=2,shuffle=False)
# for step,batch in enumerate(dataloader):
#     print(step)
#     print(batch)

#设计模型类
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear1=torch.nn.Linear(1,1,bias=True)
    def forward(self, x):
        x=self.linear1(x)
        return x

#test model
# model=LinearModel()
# dummy_input=torch.tensor([1], dtype=torch.float)
# dummy_output=model(dummy_input)
# print(dummy_output)

#创建损失函数和优化器
model=LinearModel()
dataset=MyDataset(xs, ys)
dataloader=DataLoader(dataset,batch_size=1,shuffle=True)

criterion=torch.nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(), lr=lr)

#训练
for epoch in range(max_epochs):
    for step,batch in enumerate(dataloader):
        batch_x,batch_y=batch
        batch_x=batch_x.float()
        batch_y=batch_y.float()
        pred_y=model(batch_x)
        loss=criterion(pred_y,batch_y)

        optimizer.zero_grad()#清空上一轮的梯度
        loss.backward()#反向传递参数
        optimizer.step()#更新参数

        if step%1==0:
            loss=loss.cpu().item()#把cpu里的loss值一步步转换出来
            print("Epoch:{},Step:{},Loss:{:.4f}".format(epoch, step, loss))

#test model
pred_x=torch.tensor([5],dtype=torch.float)
pred_y=model(pred_x)
print(pred_y)