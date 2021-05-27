#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from Activations import *
from Exceptions import ShapeException, LayerException


# In[2]:


layers_counter = {
}


# In[3]:


class Layer:
    '''
    Родительский класс, от него наследуются все классы.
    Может быть вызван (callable)
    Передает вход и выход, не совершая над ними никаких действий
    '''
    
    __slots__ = 'input_size', 'output_size', 'name'
    
    def __init__(self, input_size=1, output_size=1, name='Layer'):
        self.input_size = input_size
        self.output_size = output_size
        self.init_free_name(name)
        self.build(input_size)
        pass
    
    def forward_pass(self, x):
        return x
    
    def backprop(self, input_grad):
        '''
        Нет обучения на слое
        '''
        return input_grad
    
    def init_free_name(self, name):
        num = layers_counter.get(name, 0)
        self.name = name + '_{}'.format(num)
        layers_counter[name] = num + 1
        return self.name
    
    def build(self, input_size=1):
        if input_size != self.input_size:
            self.input_size = input_size
            self.build_neurons()
        pass
    
    def build_neurons(self):
        pass
    
    def __call__(self, x, *args, **kwargs):
        return self.forward_pass(x)


# In[4]:


class InputLayer(Layer):
    '''
    Передает вход и выход, не совершая над ними никаких действий
    '''
    def __init__(self, input_size=1, name='InputLayer'):
        super().__init__(input_size=input_size, output_size=input_size, name=name)
        pass


# In[5]:


class Dense(Layer):
    __slots__ = 'W', 'b', 'input', 'Z', 'activation', 'grad'
    
    def __init__(self, output_size=1, name='Dense', activation=None):
        self.activation = activation
        if activation is None:
            self.activation = Step()
        super().__init__(output_size=output_size, name=name)
        pass
    
    def build_neurons(self):
        '''
        Генерируем W из нормального распределенияя
        W ~ N(0, 1/sqrt(n)), n - число нейронов
        b ~ N(0, 1)
        '''
        sigma = int(1 / np.sqrt(self.output_size))
        self.W = np.random.normal(0, sigma, (self.input_size, self.output_size))
        self.b = np.random.normal(0, 1, self.output_size)
        pass
    
    def forward_pass(self, x):
        self.input = x
        '''
        Проброс вперед 
        -> f(xW + b) = f(Z)
        '''
        Z = np.dot(x, self.W) + self.b
        self.Z = Z
        return self.activation(Z)
    
    def backprop(self, X=np.array([])):
        return self.backprop(self.forward_pass(X))
    
    def backprop(self, input_grad):
        '''
        Проброс назад
        '''
        # dZ/dw
        self_grad = self.activation.grad(np.dot(self.input, self.W) + self.b)
        if len(self_grad.shape) > 2:
            db = np.array([np.dot(inp, s_grad) for inp, s_grad in zip(input_grad, self_grad)])
        else:
            db = input_grad * self_grad
        dW = np.dot(self.input.T, db)
        dX = np.dot(db, self.W.T)
        
        db = np.sum(db, axis=0)
        return dW, db, dX

