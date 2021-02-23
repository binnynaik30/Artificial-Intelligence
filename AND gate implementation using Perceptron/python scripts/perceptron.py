# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 12:20:13 2021

@author: Binny
"""
import numpy as np

class Perceptron(object):
    
    def __init__(self,no_of_inputs,threshold=10,learning_rate=0.005):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)
    
    def predict(self,inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
            activation = 1
        else:
            activation = 0
            
        return activation
        
    def train(self, training_inputs, labels):
        for i in range(1, self.threshold + 1):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] +=  self.learning_rate * (label - prediction)
                
            print("For epoch" + str(i) + ":")
            print("Bias value calculated = ", self.weights[0])
            print("Weights value calculated = ", self.weights[1:])
            print()