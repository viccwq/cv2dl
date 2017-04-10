# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 14:24:36 2017

@author: VIC
"""
import numpy as np
import matplotlib.pyplot as plt


#y=x^4的
#原函数
def f_fun(x):
    y = x * x * x * x
    return y
#梯度函数表达式
def g_fun(x):
    y = 3 * x * x * x
    return y
#固定学习率梯度下降（不同学习率对比）   
def grad_desc_fixed():
    array_i = np.arange(0,1000)
    array_x = np.zeros(1000, dtype=np.double,)
    #学习率
    a = 0.01
    x = 2.0
    plt.figure()
    array_x[0] = x
    for i in array_i[1:]:
        g = g_fun(x)
        x = x - a * g
        array_x[i] = x    
    plt.plot(array_i, array_x, 'r-', label='a=0.01', linewidth=2)  
    #学习率
    a = 0.04
    x = 2.0
    array_x[0] = x
    for i in array_i[1:]:
        g = g_fun(x)
        x = x - a * g
        array_x[i] = x
    plt.plot(array_i, array_x, 'b-', label='a=0.04', linewidth=2)
    plt.xlabel("time") 
    plt.ylabel("x") 
    plt.title("gradient descent fixed rate")
    plt.legend()
    plt.savefig("../screenshot/grad_desc_fixed.jpg")  
    plt.show()    
    return   

#x:当前值
#g:梯度
#a:学习率
def get_a_Armijo(x, g, a):
    c1 = 0.2
    y_now = f_fun(x)
    y_next = f_fun(x - g * a)
    
    count = 30
    while (y_now > y_next):
        a *= 2
        y_next = f_fun(x - a * g)
        count -= 1
        if (0 == count):
            break
        
    count = 50
    while (y_now - y_next < (c1 * a * g * g)):
        a /= 2
        y_next = f_fun(x - a * g)
        count -= 1
        if (0 == count):
            break
    return a
#回溯线性搜索(与固定学习率比较)
#back line search
def grad_desc_back():
    array_i = np.arange(0,60)
    array_x = np.zeros(60, dtype=np.double,)
    #学习率
    a = 0.04
    x = 2.0
    plt.figure()
    array_x[0] = x
    for i in array_i[1:]:
        g = g_fun(x)
        x = x - a * g
        array_x[i] = x    
    plt.plot(array_i, array_x, 'r-', label='fixed', linewidth=1)  
    #学习率
    a = 0.04
    x = 2.0
    array_x[0] = x
    for i in array_i[1:]:
        g = g_fun(x)
        a = get_a_Armijo(x, g, a)
        x = x - a * g
        array_x[i] = x
    plt.plot(array_i, array_x, 'b-', label='Armijo', linewidth=1)
    plt.xlabel("time") 
    plt.ylabel("x") 
    plt.title("gradient descent Aemijo")
    plt.legend()
    plt.savefig("../screenshot/grad_desc_Armijo.jpg")  
    plt.show()    
    return   
    
if __name__ == '__main__':
    
    print("Welcome to My blog!")
    grad_desc_fixed()
    grad_desc_back()
