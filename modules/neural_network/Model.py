#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np


# In[ ]:


from Layers import Layer, InputLayer
from Exceptions import ModelException
from Optimizers import Optimizer
from Loss import Loss


# In[ ]:


def batch(X, batch_size=1):
    if len(X.shape) == 2:
        X = X[:batch_size * (X.shape[0] // batch_size)]
        return np.reshape(X, (X.shape[0] // batch_size, batch_size, X.shape[1]))
    return X


# In[2]:


class Sequential():
    __slots__ = 'layer_stack', 'optimizer', 'compiled', 'name', 'loss'
    
    def __init__(self, name='Model'):
        self.layer_stack = []
        self.compiled = False
        self.name = name
        pass
    
    def add(self, layer):
        '''
        Добавления слоя в модель
        '''
        if not issubclass(type(layer), Layer):
            raise ModelException('Expected Layer, passed {}.'.format(layer))
        if not self.layer_stack:
            if layer.input_shape is None:
                raise ModelException('First layer of model must be defined on {}.'.format(layer))
            self.layer_stack.append(layer)
        else:
            if layer.input_shape is not None:
                if layer.input_shape != self.layer_stack[-1].output_shape:
                    raise ModelException("Can't add layer. Last shape layers was {}, trying to add layer with shape {}.".
                                        format(self.layer_stack[-1].output_shape, layer.input_shape))
                else:
                    layer.build()
                    self.layer_stack.append(layer)
            else:
                layer.build(new_input=self.layer_stack[-1].output_shape)
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
        self.compiled = False
        if not issubclass(type(optimizer), Optimizer):
            raise ModelException('Unable to compile: expect Optimizer, get {}'.
                                format(optimizer))
        if not issubclass(type(loss), Loss):
            raise ModelException('Unable to compile: expect Loss, get {}'.
                                format(loss))
        if not self.layer_stack:
            raise ModelException('Unable to compile: no layers. Add layers with funtion model.add.')
        if len(self.layer_stack) > 1:
            for i in range(len(self.layer_stack) - 1):
                if self.layer_stack[i].output_shape != self.layer_stack[i+1].input_shape:
                    raise ModelException('Unable to compile: output share on {} is {} but input layer on {} is {}.'.
                                        format(self.layer_stack.name, self.layer_stack.output_shape, 
                                              self.layer_stack.name, self.layer_stack.input_shape))
        try:
            for layer in self:
                layer.build()
        except Exception as e:
            raise ModelException('Unable to build layers on compile: {}', e.strerror)
        self.optimizer = optimizer
        self.loss = loss
        self.compiled = True
        pass
    
    def fit(self, X_train, y_train, batch_size=10, verbose=False):
        if not self.compiled:
            raise ModelException("Unable to fit: Model not compiled.")
        history = []
        if len(y_train.shape) == 1:
            # Преобразуем данные в двумерный массив
            y_train = y_train.reshape((-1, 1))
        X = batch(X_train, batch_size=batch_size)
        Y = batch(y_train, batch_size=batch_size)
        indexes = np.array(range(len(X)))
        for epoch in self.optimizer:
            np.random.shuffle(indexes)
            for idx in indexes:
                for x_, y_ in zip(X[idx], Y[idx]):
                    x = x_.reshape((1, -1))
                    y = y_.reshape((1, -1))
                    # Forward pass
                    y_pred = self.predict(x)
                    # Backprop
                    dX = self.loss.grad(y, y_pred)
                    for layer in self.layer_stack[::-1]:
                        if isinstance(layer, InputLayer):
                            continue
                        dX = self.optimizer.step_backprop(dX, layer)
            y_pred = self.predict(X_train)
            loss = np.sum(self.loss.f(y_train, y_pred)) / len(y_train)
            history.append(loss)
        return np.array(history)
    
    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            y_pred.append(x)
            for layer in self.layer_stack:
                y_pred[-1] = layer(y_pred[-1])
        y_pred = np.array(y_pred)
        if y_pred.shape[1] == 1:
            y_pred = np.reshape(y_pred, (-1, y_pred.shape[-1]))
        return y_pred
    
    def __call__(self, X_test, *args, **kwargs):
        return self.predict(X_test)
    
    def __iter__(self):
        for layer in self.layer_stack:
            yield layer
        pass

