#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from Exceptions import OptimizerException


# In[2]:


class Penalty:
    __slots__ = 'coef'
    
    def __init__(self, coef=0):
        self.coef = coef
        pass
    
    def get_penalty(self, layer, teta):
        return 0
    
    def __call__(self, layer, teta, *args, **kwargs):
        return self.get_penalty(layer, teta)


# In[3]:


class L1(Penalty):
    def __init__(self, coef=0.1):
        '''
        coef - koefficient before l1 regulizer
        '''
        super().__init__(coef=coef)
        pass
    
    def get_penalty(self, layer, teta):
        '''
        l1 regularization
        '''
        return - (teta * self.coef) * np.sign(layer.W)


# In[4]:


class L2(Penalty):
    def __init__(self, coef=0.1):
        '''
        coef - koefficient before l2 regulizer
        '''
        super().__init__(coef=coef)
        pass
    
    def get_penalty(self, layer, teta):
        '''
        l2 regularization
        '''
        return - (teta * self.coef) * layer.W


# In[5]:


class Optimizer:
    __slots__ = 'teta_0', 'K', 'epochs', 'teta_k', 'teta', 'penalties'
    
    def __init__(self, teta_0=1e-3, teta_k=1e-5, K=500, epochs=1000, clipvalue=1e7, penalties=[]):
        '''
        teta_i = (1 - i/K)*teta_0 + 1/K * teta_k, i < k
        teta_i = teta_k, i >= k
        100 <= K <= 1000
        '''
        for penalty in penalties:
            if not issubclass(type(penalty), Penalty):
                raise OptimizerException('Expect Penalty, got: ', penalty)
        self.penalties = penalties
        self.clipvalue = clipvalue
        self.teta_0 = teta_0
        self.teta = teta_0
        self.K = K
        self.teta_k = teta_k
        self.epochs = epochs
        pass
    
    def clip(self, grad):
        if self.clipvalue is not None:
            grad[grad > self.clipvalue] = self.clipvalue
            grad[grad < -self.clipvalue] = -self.clipvalue
        return grad
        
    def step_backprop(self, grad_x, layer, n_obj=1):
        dW, db, dX = layer.backprop(grad_x, n_obj=n_obj)
        return dX
    
    def __iter__(self):
        for epoch in range(self.epochs):
            self.next_iter(epoch)
            yield epoch
            
    def next_iter(self, epoch):
        self.teta = self.teta_k
        if epoch < self.K:
            self.teta = (1. - epoch/self.K)*self.teta_0 + epoch/self.K*self.teta_k
            
    def __call__(self, grad_x, layer, n_obj=1, *args, **kwargs):
        return self.step_backprop(grad_x, layer, n_obj=n_obj)


# In[6]:


class SGD(Optimizer):
    def __init__(self, teta_0=1e-3, teta_k=1e-5, K=500, epochs=1000, clipvalue=1e7, penalties=[]):
        super().__init__(teta_0=teta_0, teta_k=teta_k, K=K, epochs=epochs, clipvalue=clipvalue, penalties=penalties)
        pass
    
    def step_backprop(self, grad_x, layer, n_obj=1):
        '''
        Include l1, l2 regularization
        '''
        dW, db, dX = layer.backprop(grad_x)
        W = layer.W
        b = layer.b
        teta = self.teta / n_obj
        delta_W = teta * dW
        for penalty in self.penalties:
            delta_W += penalty(layer, teta)
        layer.W -= delta_W
        layer.b -= teta * db
        return dX
    
    def __call__(self, grad_x, layer, n_obj=1, *args, **kwargs):
        clipped_grad = self.clip(grad_x)
        return self.step_backprop(clipped_grad, layer, n_obj=n_obj)


# In[7]:


class MomentumSGD(Optimizer):
    __slots__ = 'momontum_w', 'momentum_b', 'a'
    
    def __init__(self, teta_0=1e-3, teta_k=1e-5, K=500, epochs=1000, clipvalue=1e7, penalties=[], a=0.9):
        super().__init__(teta_0=teta_0, teta_k=teta_k, K=K, epochs=epochs, clipvalue=clipvalue, penalties=penalties)
        self.momentum_w = {}
        self.momentum_b = {}
        self.a = a
        pass
    
    def step_backprop(self, grad_x, layer, n_obj=1):
        '''
        Include l1, l2 regularization
        '''
        dW, db, dX = layer.backprop(grad_x)
        teta = self.teta / n_obj
        
        self.momentum_w[layer] = self.a * self.momentum_w.get(layer, 0) - teta * dW
        self.momentum_b[layer] = self.a * self.momentum_b.get(layer, 0) - teta * db
        
        delta_W = teta * self.momentum_w[layer]
        for penalty in self.penalties:
            delta_W += penalty(layer, teta)
        
        layer.W += delta_W
        layer.b += teta * self.momentum_b[layer]
        return dX

