# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 13:24:09 2022

@author: 95287
"""

######通过对最小二乘求导、自己编写梯度下降算法和调用sklearn中线性回归库，分别对pga数据进行单变量回归实验，可以发现
#1、通过求导加最小二乘法算得的误差最小，为0.313810760011419，python中默认精度到小数点你后15位
#2、调用sklearn中线性模型库优化求解也能达到最小得到的值为0.31381076001141883，（sklearn中最小二乘优化采用scipy中的optimize库进行最小二乘优化，收敛阈值设置为1e-10）
#3、通过自己编写的梯度下降方法获得的值为0.31381494512654245误差最大，可能与学习率和收敛阈值设置有关
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
# 导入测试数据
pga = pd.read_csv("pga.csv")

# Normalize the data 归一化值 (x - mean) / (std)
pga.distance = (pga.distance - pga.distance.mean()) / pga.distance.std()
pga.accuracy = (pga.accuracy - pga.accuracy.mean()) / pga.accuracy.std()
'''
#输出归一化后图像
plt.scatter(pga.distance, pga.accuracy)
plt.xlabel('normalized distance')
plt.ylabel('normalized accuracy')
plt.show()
'''


from sklearn.linear_model import LinearRegression
import numpy as np

# We can add a dimension to an array by using np.newaxis
print("Shape of the series:", pga.distance.shape)
print("Shape with newaxis:", pga.distance[:, np.newaxis].shape)

# The X variable in LinearRegression.fit() must have 2 dimensions
lm = LinearRegression()
lm.fit(pga.distance[:, np.newaxis], pga.accuracy)
###获取训练好的模型的系数和截距
theta1 = lm.coef_[0]
theta0 = lm.intercept_
print (theta1)
print(theta0)
###输出误差（residues)
print(lm._residues/(2*len(pga.distance)))

'''
##########################单变量情况下####################
#实现方法一（直接最小二乘法求导）
#单变量，最小二乘法
def calcAB(x,y):
    n = len(x)
    sumX,sumY,sumXY,sumXX =0,0,0,0
    for i in range(0,n):
        sumX  += x[i]
        sumY  += y[i]
        sumXX += x[i]*x[i]
        sumXY += x[i]*y[i]
    a = (n*sumXY -sumX*sumY)/(n*sumXX -sumX*sumX)
    b = (sumXX*sumY - sumX*sumXY)/(n*sumXX-sumX*sumX)
    #最终损失
    co=sum(vi**2 for vi in y-(a*x+b))/(2*n)
    return a,b,co
xi = pga.distance
yi = pga.accuracy
a,b,co=calcAB(xi,yi)
print("y = %10.5fx + %10.5f" %(a,b))
print(co)
x = np.linspace(0,10)
y = a * x + b
plt.plot(x,y)
plt.scatter(xi,yi)
plt.show()
'''

'''
#实现方法二（定义代价函数后，用梯度下降法求最优）
#代价函数
def cost(theta0, theta1, x, y):
    # Initialize cost
    J = 0
    m = len(x)
    # 通过每次观察进行循环
    for i in range(m):
        # 计算假设，在y=ax+b中，theta1相当于a，theta0相当于b
        h = theta1 * x[i] + theta0
        # Add to cost
        J += (h - y[i])**2
    # Average and normalize cost
    J /= (2*m)
    return J


#只考虑theta1的情况下，可视化代价
# The cost for theta0=0 and theta1=1
print(cost(0, 1, pga.distance, pga.accuracy))

theta0 = 0
#即以5/100作为learning rate
theta1s = np.linspace(-3,2,100)
costs = []
for theta1 in theta1s:
    costs.append(cost(theta0, theta1, pga.distance, pga.accuracy))
#输出最损失(梯度下降无法得到最优损失？)
print(min(costs))
plt.plot(theta1s, costs)
plt.show()

#同时考虑theta0,theta1和cost可以构建三维图，可视化代价
from mpl_toolkits.mplot3d import Axes3D
# Use these for your excerise 
theta0s = np.linspace(-2,2,100)
theta1s = np.linspace(-2,2, 100)
COST = np.empty(shape=(100,100))
# Meshgrid for paramaters 
T0S, T1S = np.meshgrid(theta0s, theta1s)
# for each parameter combination compute the cost
for i in range(100):
    for j in range(100):
        COST[i,j] = cost(T0S[0,i], T1S[j,0], pga.distance, pga.accuracy)
# make 3d plot
fig2 = plt.figure()
ax = fig2.add_subplot(projection='3d')
ax.plot_surface(X=T0S,Y=T1S,Z=COST)
plt.show()

# 对 theta1 进行求导
def partial_cost_theta1(theta0, theta1, x, y):
    # Hypothesis即模型
    h = theta0 + theta1*x
    # Hypothesis minus observed times x
    diff = (h - y) * x
    # Average to compute partial derivative
    partial = diff.sum() / (x.shape[0])
    #求得关于theta1的梯度
    return partial
#求得模型在5，0处的梯度
partial1 = partial_cost_theta1(5, 0, pga.distance, pga.accuracy)
print("partial1 =", partial1)

# 对theta0 进行求导
# Partial derivative of cost in terms of theta0
def partial_cost_theta0(theta0, theta1, x, y):
    # Hypothesis
    h = theta0 + theta1*x
    # Difference between hypothesis and observation
    diff = (h - y)
    # Compute partial derivative
    partial = diff.sum() / (x.shape[0])
    #返回关于theta0的梯度
    return partial

partial0 = partial_cost_theta0(1, 1, pga.distance, pga.accuracy)
print("partial0 =", partial0)

# x is our feature vector -- distance
# y is our target variable -- accuracy
# alpha is the learning rate
# theta0 is the intial theta0 
# theta1 is the intial theta1
def gradient_descent(x, y, alpha=0.1, theta0=0, theta1=0):
    max_epochs = 1000 # Maximum number of iterations 最大迭代次数
    counter = 0       # Intialize a counter 当前第几次
    c = cost(theta1, theta0, pga.distance, pga.accuracy)  ## Initial cost 当前代价函数
    costs = [c]     # Lets store each update 每次损失值都记录下来
    # Set a convergence threshold to find where the cost function in minimized
    # When the difference between the previous cost and current cost 
    #        is less than this value we will say the parameters converged
    # 设置一个收敛的阈值 (两次迭代目标函数值相差没有相差多少,就可以停止了)
    convergence_thres = 0.000001  
    cprev = c + 10   #+10避免while循环时cprev和c有初始差，一定要比初始代价高
    theta0s = [theta0]
    theta1s = [theta1]

    # When the costs converge or we hit a large number of iterations will we stop updating
    # 两次间隔迭代目标函数值相差没有相差多少(说明可以停止了)
    while (np.abs(cprev - c) > convergence_thres) and (counter < max_epochs):
        cprev = c
        # Alpha times the partial deriviative is our updated
        # 先求导, 导数相当于步长
        update0 = alpha * partial_cost_theta0(theta0, theta1, x, y)
        update1 = alpha * partial_cost_theta1(theta0, theta1, x, y)

        # Update theta0 and theta1 at the same time
        # We want to compute the slopes at the same set of hypothesised parameters
        #             so we update after finding the partial derivatives
        # -= 梯度下降，+=梯度上升
        theta0 -= update0
        theta1 -= update1
        
        # Store thetas
        theta0s.append(theta0)
        theta1s.append(theta1)
        
        # Compute the new cost
        # 当前迭代之后，参数发生更新  
        c = cost(theta0, theta1, pga.distance, pga.accuracy)

        # Store updates，可以进行保存当前代价值
        costs.append(c)
        counter += 1   # Count
        
    # 将当前的theta0, theta1, costs值都返回去
    return {'theta0': theta0, 'theta1': theta1, "costs": costs}

print("Theta0 =", gradient_descent(pga.distance, pga.accuracy)['theta0'])
print("Theta1 =", gradient_descent(pga.distance, pga.accuracy)['theta1'])
print("costs =", gradient_descent(pga.distance, pga.accuracy)['costs'])

descend = gradient_descent(pga.distance, pga.accuracy, alpha=.01)
plt.scatter(range(len(descend["costs"])), descend["costs"])
plt.show()
'''
