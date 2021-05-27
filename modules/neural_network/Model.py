#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.model_selection import train_test_split


# In[2]:


from Layers import Layer, InputLayer
from Exceptions import ModelException
from Optimizers import Optimizer
from Loss import Loss


# In[3]:


model_names = {
}


# In[4]:


def batch(X, batch_size=1):
    if len(X.shape) == 2:
        X = X[:batch_size * (X.shape[0] // batch_size)]
        return np.reshape(X, (X.shape[0] // batch_size, batch_size, X.shape[1]))
    return X


# In[5]:


class Sequential():
    __slots__ = 'layer_stack', 'optimizer', 'name', 'loss', 'eps'
    
    def __init__(self, name='Model', eps=1e-10):
        self.eps = eps
        self.layer_stack = []
        self.init_free_name(name)
        pass
    
    def init_free_name(self, name):
        num = model_names.get(name, 0)
        self.name = name + '_{}'.format(num)
        model_names[name] = num + 1
        return self.name
    
    def add(self, layer):
        '''
        Добавления слоя в модель
        слой должен наследовать Layer
        '''
        if not issubclass(type(layer), Layer):
            raise ModelException('Expect Layer, passed {}.'.format(layer))
        if not self.layer_stack:
            if not issubclass(type(layer), InputLayer):
                raise ModelException('Expect InputLayer, passed {}'.format(layer))
        else:
            layer.build(input_size=self.layer_stack[-1].output_size)
        self.layer_stack.append(layer)
        pass
        
    def summary(self):
        # PRINT
        print('+' + '-' * 26 + '-' + '-' * 18 + '-' + '-' * 18 + '+')
        print('| {:^62} |'.format(self.name))
        print('+' + '-' * 26 + '+' + '-' * 18 + '+' + '-' * 18 + '+')
        print('| {:^24} | {:^16} | {:^16} |'.format('Layer name', 'Input shape', 'Output shape'))
        print('+' + '-' * 26 + '+' + '-' * 18 + '+' + '-' * 18 + '+')
        for layer in self.layer_stack:
            print('| {:^24} | {:^16} | {:^16} |'.format(layer.name[:24],
                                                        layer.input_shape,
                                                        layer.output_shape))
        print('+' + '-' * 26 + '+' + '-' * 18 + '+' + '-' * 18 + '+')
        pass
    
    def compile(self, optimizer, loss):
        if not issubclass(type(optimizer), Optimizer):
            raise ModelException('Unable to compile: expect Optimizer, get {}'.format(optimizer))
        if not issubclass(type(loss), Loss):
            raise ModelException('Unable to compile: expect Loss, get {}'.format(loss))
        self.optimizer = optimizer
        self.loss = loss
        pass
    
    def fit(self, X_train, y_train, batch_size=10, verbose=False, n_verbose=10, task="classification"):
        max_exp = 1. / self.eps
        
        if not self.optimizer or not self.loss:
            raise ModelException('Unable to fit: Model not compiled.')
        history = []
        if len(y_train.shape) == 1:
            # Преобразуем данные в двумерный массив
            y_train = y_train.reshape((-1, 1))
        stratify = None
        if task != "classification":
            stratify = y_train
        X_tr, X_control, y_tr, y_control = train_test_split(X_train, y_train, stratify=stratify)
        X = batch(X_tr, batch_size=batch_size)
        Y = batch(y_tr, batch_size=batch_size)
        indexes = np.array(range(len(X)))
        if verbose:
            print('Start fit.')
        for epoch in self.optimizer:
            np.random.shuffle(indexes)
            for num_iter, idx in enumerate(indexes):
                x, y = X[idx], Y[idx]
                # Forward pass
                y_pred = self.predict(x)
                # Backprop
                # dX = np.sum(self.loss.grad(y, y_pred), axis=0)
                dX = self.loss.grad(y, y_pred)
                # Ограничиваем сверху и снизу
                dX = np.where(np.abs(dX) > max_exp, np.sign(dX) * max_exp, dX)
                n_obj = y.shape[0]
                for layer in self.layer_stack[::-1]:
                    if isinstance(layer, InputLayer):
                        continue
                    dX = self.optimizer(dX, layer, n_obj=n_obj)
                if verbose and num_iter % n_verbose == 0:
                    print('epoch\t{}\t|\titer\t{}\t/\t{}\t'.format(epoch, num_iter, len(X)), end='\r')
            y_pred = self.predict(X_control)
            loss = np.mean(self.loss.f(y_control, y_pred))
            history.append(loss)
        if verbose:
            print('    '*8, end='\r')
            print('\rEnd fit.')
        return np.array(history)
    
    def predict(self, X_test):
        y_pred = X_test
        for layer in self.layer_stack:
            y_pred = layer(y_pred)
        return y_pred
    
    def __call__(self, X_test, *args, **kwargs):
        return self.predict(X_test)
    
    def __iter__(self):
        for layer in self.layer_stack:
            yield layer
        pass

