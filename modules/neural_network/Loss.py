#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from Exceptions import LossException


# In[2]:


class FuncLoss:
    def __init__(self):
        pass
    
    def f(self, y_true, y_pred):
        pass
    
    def __call__(self, y_true, y_pred, *args, **kwargs):
        return self.f(y_true, y_pred)


# In[3]:


class Loss(FuncLoss):
    def __init__(self):
        pass
    
    def f(self, y_true, y_pred):
        pass
    
    def grad(self, y_true, y_pred):
        pass
    
    def __call__(self, y_true, y_pred, *args, **kwargs):
        return self.f(y_true, y_pred)


# In[4]:


# Classification
class CategoricalEntropy(Loss):
    '''
    Категориальная энтропия H(y_true, y_pred) = -sum_i y_true_i log(y_pred_i)
    dH/dy_pred_j = -d/dy_pred_j sum_i y_true_i log(y_pred_i) =
    = -d/dy_pred_j y_true_j log(y_pred_j) = -y_true_j / y_pred_j
    '''
    def __init__(self, eps=1e-6):
        self.eps = eps
        
    def f(self, y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred+self.eps), axis=1)
    
    def grad(self, y_true, y_pred):
        return -y_true / (y_pred+self.eps)


# In[5]:


# Classification
class SparseCategoricalEntropy(Loss):
    '''
        Категориальная энтропия H(p, q), где p не является onehot вектором.
        dH/dq_j = - d/dq_j log(q_i)_[p_j] = - d/dq_j log(q_j)_[p_j] = (-1/q_j)_[p_j]
    '''
    def __init__(self, eps=1e-6):
        self.eps = eps

    def f(self, y_true, y_pred):
        return -np.array([np.log(y_pred+self.eps)[idx] for idx in list(enumerate(y_true))])

    def grad(self, y_true, y_pred):
        J = np.zeros_like(y_pred)
        for i, y in enumerate(y_true):
            J[i, y] = -1.0/(y_pred[i, y]+self.eps)
        return J


# In[6]:


class MSE(Loss):
    '''
    Среднеквадратичная ошибка
    Mean Square Error
    H(y_true, y_pred) = sum_i (y_pred_i - y_true_i)^2
    dH/dy_pred_j = 2*(y_pred_i - y_true_i)
    '''
    def __init__(self):
        pass
        
    def f(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))
    
    def grad(self, y_true, y_pred):
        return 2 * (y_pred - y_true)


# In[7]:


class MAE(Loss):
    '''
    Абсолютная средняя ошибка
    Mean Absolute error, L1 Loss
    H(y_true, y_pred) = sum_i |y_pred_i - y_true_i|
    dH/dy_pred_j = 1 (y_pred_i >= 0), или -1 (y_pred_i < 0)
    Параметр eps - константа для численной стабильности (сравнение с нулем)
    '''
    def __init__(self, eps=1e-12):
        self.eps = eps
        pass
        
    def f(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))
    
    def grad(self, y_true, y_pred):
        return -np.sign(y_pred)


# In[8]:


class SMAE(Loss):
    '''
    Huber Loss, Smooth Mean Absolute Error
    H(y_true, y_pred) = (y_true - y_pred)^2 / 2 (|y-f(z)| <= beta), или 
        beta * |y_true - y_pred| - beta^2 / 2
    dH/dy_pred_j = y - y_pred (|y-f(z)| <= beta), или 
        beta (y >= 0), -beta (y < 0)
    '''
    
    def __init__(self, beta=1):
        self.beta = beta
        pass
    
    def f(self, y_true, y_pred):
        diff = y_true - y_pred
        return np.where(np.abs(diff) <= self.beta, 
                       diff * diff / 2.,
                       self.beta * np.abs(diff) - self.beta**2 / 2.)
        
    def grad(self, y_true, y_pred):
        diff = y_true - y_pred
        return np.where(np.abs(diff) <= self.beta, 
                       - diff,
                       - np.sign(diff) * self.beta)


# In[9]:


class ZeroOne(FuncLoss):
    '''
    0-1 loss function
    for classification
    H(y_true, y_pred) = y_true == y_pred
    H' undefined
    '''
    def __init__(self, eps=1e-7):
        self.eps = eps
        pass
    
    def f(self, y_true, y_pred):
        return np.where(np.abs(y_true - y_pred) < self.eps, 1, 0)


# In[10]:


class SimpleLoss(Loss):
    '''
    Simple loss
    H(y_true, y_pred) = y_true - y_pred
    dH/dy_pred = -1
    '''
    def __init__(self, eps=1e-7):
        self.eps = eps
        pass
    
    def f(self, y_true, y_pred):
        return y_true - y_pred
    
    def grad(self, y_true, y_pred):
        return - np.ones_like(y_true)


# In[11]:


class LogLoss(Loss):
    '''
    log_loss
    H(y_true, y_pred)  = - (y_true ln(y_pred) + (1 - y_true)ln(1 - y_pred))
    dH/d_y_pred = (y_pred - y_true)/(y_pred-y_pred^2)
    '''
    def __init__(self, beta = 1e7, eps = 1e-7):
        self.beta = beta
        self.eps = eps
        pass
        
    def f(self, y_true, y_pred):
        y_pred = np.where(y_pred < self.eps, self.eps, y_pred)
        y_pred = np.where(1. - y_pred < self.eps, 1. - self.eps, y_pred)
        lg = np.log(y_pred)
        lg1 = np.log(1. - y_pred)
        return - (y_true * lg + (1. - y_true) * lg1)
    
    def grad(self, y_true, y_pred):
        divisor = y_pred * (1 - y_pred)
        divisor = np.where(np.abs(divisor) < self.eps, divisor + self.eps, divisor)
        return (y_pred - y_true) / divisor


# In[12]:


loss_alias = {
    'ce': CategoricalEntropy(),
    'sce': SparseCategoricalEntropy(),
    'mse': MSE(),
    'mae': MAE(),
    'smae': SMAE(),
    'log_loss': LogLoss(),
}


# In[13]:


def register_loss(alias, loss):
    '''
        Добавляет alias: loss в словарь loss_alias
        :param alias: имя функции ошибки
        :param loss: объект ошибки
    '''
    if not issubclass(loss, FuncLoss):
        raise LossException('{} must be inherit from {}.'.format(loss, FuncLoss))
    loss_alias[alias] = loss

