import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

lr=1e-2
epochs=300
max_norm=3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义二次函数参数
a, b, c = 1, -2, 1  # f(x) = x^2 - 2x + 1
x_train = np.random.choice(np.linspace(-10, 10, 100), size=20, replace=False)# replace参数，是否取相同值
y_train = a * x_train ** 2 + b * x_train + c + np.random.normal(0, 2, x_train.shape)# 均值，标准差，规模

# 绘制真实函数
x_plot = np.linspace(-10, 10, 100)
y_plot = a * x_plot ** 2 + b * x_plot + c
plt.scatter(x_train, y_train, color='red', label='Training Data')
plt.plot(x_plot, y_plot, label='True Function', color='blue')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.title('Quadratic Function and Training Data')
plt.show()

class MyDataset(Dataset):
    def __init__(self, xs, ys):
        super(MyDataset, self).__init__()
        self.x_tensor = torch.tensor(xs, dtype=torch.float32).view(-1, 1).to(device)
        self.y_tensor = torch.tensor(ys, dtype=torch.float32).view(-1, 1).to(device)

    def __len__(self):
        return self.x_tensor.shape[0]

    def __getitem__(self, index):
        return self.x_tensor[index], self.y_tensor[index]

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1,1000),
            nn.ReLU(),
            nn.Linear(1000,1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000,1),
            nn.ReLU(),
        )

    def forward(self, x):
        x=self.net(x)
        x=x.view(-1, 1)
        return x

# 定义模型和数据加载器
model = MyNet().to(device)
dataset = MyDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

# 训练
for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
        batch_x, batch_y = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()

        pred_y = model(batch_x)
        loss = criterion(pred_y, batch_y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()

# 创建测试数据
x_test = np.linspace(-10, 10, 100)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32).view(-1, 1).to(device)

# 使用训练好的模型进行预测
model.eval().to(device)
with torch.no_grad():
    y_test_pred = model(x_test_tensor).cpu().numpy()

# 绘制测试结果
plt.scatter(x_train, y_train, color='red', label='Training Data')
plt.plot(x_plot, y_plot, label='True Function', color='blue')
plt.plot(x_test, y_test_pred, label='Predicted Function', color='green')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.title('Model Prediction on Test Data')
plt.text(-10, 15, f'Epochs: {epochs}', fontsize=12, color='black')
plt.text(-10, 8, f'lr: {lr}', fontsize=12, color='black')
plt.text(-10, 1, f'max_norm: {max_norm}', fontsize=12, color='black')
plt.show()
