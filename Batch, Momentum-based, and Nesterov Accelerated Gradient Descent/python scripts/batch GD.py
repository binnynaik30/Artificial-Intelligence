# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 11:05:19 2021

@author: Binny
"""

import numpy as np

X = [0.75, 3.24]
Y = [0.11, 0.83]

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

def do_gradient_descent():
    w, b, lr, max_epochs = 0, 0, 0.1, 999
    for i in range(max_epochs):
        dw, db = 0, 0
        print("error= ", error(w, b))
        for x,y in zip(X,Y):
            dw += grad_w(w, b, x, y)
            db += grad_b(w, b, x, y)
        step_size_w = dw * lr
        step_size_b = db * lr
        w = w - step_size_w
        b = b - step_size_b
        print("weigh w1 = ",w)
        print("bias b = ", b)
        print("step_size_w = ", step_size_w)
        print("step_size_b = ", step_size_b)
        
        print()
        
do_gradient_descent()
