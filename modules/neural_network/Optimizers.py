#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


class Optimizer:
    __slots__ = 'learning_rate', 'epochs'
    
    def __init__(self, learning_rate=1e-3, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        
    def step_backprop(self, grad_x, layer):
        dW, db, dX = layer.backprop(grad_x)
        return dX
    
    def __iter__(self):
        for epoch in range(self.epochs):
            yield epoch


# In[3]:


class SGD(Optimizer):
    def __init__(self, learning_rate=1e-3, epochs=100):
        super().__init__(learning_rate, epochs)
        pass
        
    def step_backprop(self, grad_x, layer):
        dW, db, dX = layer.backprop(grad_x)
        layer.W -= self.learning_rate * dW
        layer.b -= self.learning_rate * db
        return dX


# In[4]:


class MoventumSGD(Optimizer):
    __slots__ = 'momentum_w', 'momentum_b', 'a'
    
    def __init__(self, learning_rate=1e-3, epochs=100, a=0.9):
        super().__init__(learning_rate, epochs)
        self.momentum_w = {}
        self.momentum_b = {}
        self.a = a
        pass
    
    def step_backprop(self, grad_x, layer):
        dW, db, dX = layer.backprop(grad_x)
        
        self.momentum_w[layer] = self.a * self.momentum_w.get(layer, 0) + self.learning_rate * dW
        self.momentum_b[layer] = self.a * self.momentum_b.get(layer, 0) + self.learning_rate * db
        
        layer.W -= self.momentum_w[layer]
        layer.b -= self.momentum_b[layer]
        return dX

