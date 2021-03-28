#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from Exceptions import LossException


# In[2]:


class Loss:
    def __init__(self):
        pass

    def f(self, y_true, y_pred):
        pass
    
    def grad(self, y_true, y_pred):
        pass
    
    def __call__(self, y_true, y_pred, *args, **kwargs):
        return self.f(y_true, y_pred)


# In[3]:


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


# In[4]:


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


# In[5]:


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


# In[10]:


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


# In[13]:


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


# In[14]:


loss_alias = {
    'ce': CategoricalEntropy(),
    'sce': SparseCategoricalEntropy(),
    'mse': MSE(),
    'mae': MAE(),
    'smae': SMAE(),
}


# In[7]:


def register_loss(alias, loss):
    """
        Добавляет alias: loss в словарь loss_alias
        :param alias: имя функции ошибки
        :param loss: объект ошибки
    """
    if not issubclass(loss, Loss):
        raise LossException('{} must be inherit from {}.'.format(loss, Loss))
    loss_alias[alias] = loss

