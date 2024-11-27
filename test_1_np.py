import numpy as np
import matplotlib.pyplot as plt

# 定义二次函数参数
a, b, c = 1, -2, 1  # f(x) = x^2 - 2x + 1
# 生成20组数据
x_train = np.random.uniform(-10, 10, 20)# 生成不重复的20个数
y_train = a * x_train**2 + b * x_train + c + np.random.normal(0, 6, x_train.shape)# 均值，标准差，规模

# 绘制真实函数
x_plot = np.linspace(-10, 10, 100)
y_plot = a * x_plot**2 + b * x_plot + c

# 绘制真实函数和训练数据对比
plt.scatter(x_train, y_train, color='red', label='Training Data')
plt.plot(x_plot, y_plot, label='True Function', color='blue')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.title('Quadratic Function and Training Data')
plt.show()

# 函数拟合，最小二乘法拟合参数
para=np.polyfit(x_train,y_train,2)

# 创建测试数据
x_test = np.linspace(-10, 10, 100)
y_test = np.polyval(para,x_test)

# 绘制测试结果
plt.scatter(x_train, y_train, color='red', label='Training Data')
plt.plot(x_plot, y_plot, label='True Function', color='blue')
plt.plot(x_test, y_test, label='Predicted Function', color='green')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.title('Model Prediction on Test Data')
plt.show()