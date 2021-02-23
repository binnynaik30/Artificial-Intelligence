# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 12:34:29 2021

@author: Binny
"""


from perceptron import Perceptron
import numpy as np

training_inputs = []
training_inputs.append(np.array([1,1]))
training_inputs.append(np.array([1,0]))
training_inputs.append(np.array([0,1]))
training_inputs.append(np.array([0,0]))

labels = np.array([1,0,0,0])

p_object = Perceptron(2)
p_object.train(training_inputs, labels)

test_input = np.array([1,1])
print("Input to model = ", test_input, end='     ,')
print("Output of trained model = ", p_object.predict(test_input))