# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 13:22:53 2021

@author: Binny
"""

import numpy as np
import time
from matplotlib import pyplot as plt


X = [0.75, 3.24]
Y = [0.11, 0.83]
w_list = []
b_list =[]
step_size_w_list = []
step_size_b_list = []

def f(w, b, x):
    return 1.0/(1.0 + np.exp(-(w*x + b)))

def error(w, b):
    err = 0.0
    for x,y in zip(X,Y):
        fx = f(w,b,x)
        err += (fx - y) ** 2
    err = err / 2
    return err
    
def grad_b(w, b, x, y):
    fx = f(w, b, x)
    return (fx - y) * fx * (1 - fx)

def grad_w(w, b, x, y):
    fx = f(w, b, x)
    return (fx - y) * fx * (1 - fx) * x

def do_nesterov_accelerated_gradient_descent():
    w, b, lr, max_epochs = 0, 0, 0.1, 84
    prev_w, prev_b, gamma = 0, 0, 0.9
    
    for i in range(max_epochs):
        dw, db = 0, 0
        
        print("error= ", error(w, b))
        
        step_size_w = gamma * prev_w
        step_size_b = gamma * prev_b
        
        for x,y in zip(X,Y):
            dw += grad_w(w - step_size_w, b - step_size_b, x, y)
            db += grad_b(w - step_size_w, b - step_size_b, x, y)
        step_size_w = gamma * prev_w + lr * dw
        step_size_b = gamma * prev_b + lr * db

        step_size_w_list.append(step_size_w)
        step_size_b_list.append(step_size_b)
        
        w = w - step_size_w
        b = b - step_size_b

        w_list.append(w)
        b_list.append(b)

        prev_w = step_size_w
        prev_b = step_size_b  
        
        print("weight w1 = ",w)
        print("bias b = ", b)
        print("step_size_w = ", step_size_w)
        print("step_size_b = ", step_size_b)
        
        print()
   
start = time.time()     
do_nesterov_accelerated_gradient_descent()
end = time.time()

print("total time by nesterov accelerated = ", end-start)

plt.xlabel("weights")
plt.ylabel("bias")
plt.plot(w_list,b_list)
plt.show()

plt.xlabel("step size of weight")
plt.ylabel("weights")
plt.plot(step_size_w_list, w_list)
plt.show()

plt.xlabel("step size of bias")
plt.ylabel("bias")
plt.plot(step_size_b_list, b_list)
plt.show()
