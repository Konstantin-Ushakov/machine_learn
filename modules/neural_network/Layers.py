#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from Shape import Shape
from Activations import *
from Exceptions import ShapeException, LayerException


# In[2]:


layers_counter = {
}


# In[3]:


class Layer:
    '''
    Родительский класс, от него наследуются все классы.
    Может быть вызван (является callable)
    '''
    
    __slots__ = 'input_shape', 'output_shape', 'name'
    
    def __init__(self, input_shape, output_shape, name):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.init_free_name(name)
        pass
    
    def build(self, new_input=None):
        pass
    
    def forward_pass(self, x):
        return x
    
    def backprop(self, input_grad):
        return input_grad
    
    def __call__(self, input, *args, **kwargs):
        return self.forward_pass(input)
    
    def init_free_name(self, name):
        # Если было дано уже используемое имя, добавляем цифру к уникальности иначе 0
        num = layers_counter.get(name, 0)
        self.name = name + '_{}'.format(num)
        layers_counter[name] = num + 1
        return self.name


# In[4]:


class InputLayer(Layer):
    '''
    Входной слой модели.
    Отсутствует обучение на данном слое.
    '''
    
    def __init__(self, input_shape, batch_size=1, name='InputLayer'):
        try:
            inp = Shape(input_shape)
            inp.change_batch_size(batch_size)
            super().__init__(inp, inp, name)
        except ShapeException as e:
            raise LayerException('Unable to create InputLayer. Exception encountered in InputLayer: {}'.format(e.strerror))
            
    def backprop(self, input_grad):
        pass


# In[5]:


class Dense(Layer):
    __slots__ = 'W', 'b', 'input', 'activation', 'grad'
    
    def __init__(self, output_shape, input_shape=None, name='Dense', activation=None):
        self.input_shape = Shape(input_shape) if input_shape else None
        self.output_shape = Shape(output_shape)
        # Инициализируем новое свободное имя
        super().init_free_name(name)
        
        if activation:
            if issubclass(type(activation), Activation):
                self.activation = activation
            else:
                raise LayerException('Passed an undefined activation function to layer {}'.format(self.name))
        else:
            self.activation = Activation()
        self.W, self.b = None, None
        pass
    
    def build(self, new_input=None):
        '''
        Инициализация матрицы весов W и свободных членов b
        '''
        if self.input_shape is None:
            if new_input is None:
                raise LayerException('Input shape of layer {} is None.'.format(self.name))
            self.input_shape = Shape(new_input)
            self.output_shape.change_batch_size(self.input_shape[0])
        # self.W = np.random.normal(size=(self.input_shape[-1], self.output_shape[-1]))
        self.W = np.random.uniform(size=(self.input_shape[-1], self.output_shape[-1]))
        self.b = np.zeros((self.input_shape[0], self.output_shape[-1]))
        pass
    
    def forward_pass(self, x):
        '''
        Проброс вперед - умножение на матрицу весов W и прибавление свободных членов
        :return: f(xW + b) = f(Z), где f - функция активации (по умолчанию f:=x=>x)
        '''
        self.input = x
        if self.W is None or self.b is None:
            self.build()
        return self.activation(np.dot(x, self.W) + self.b)
    
    def backprop(self, X=np.array([])):
        return self.backprop(self.forward_pass(X))
        
    def backprop(self, input_grad):
        '''
        проброс назад
        '''
        self_grad = self.activation.grad(np.dot(self.input, self.W) + self.b)
        if len(self_grad.shape) > 2:
            db = np.array([np.dot(inp, s_grad) for inp, s_grad in zip(input_grad, self_grad)])
        else:
            db = input_grad * self_grad
        if len(self.input.shape) < 2:
            self.input = np.reshape(self.input, (1, self.input.shape[0]))
        dW = np.dot(self.input.T, db)
        dX = np.dot(db, self.W.T)
        return dW, db, dX
    
    def __call__(self, input, *args, **kwargs):
        return self.forward_pass(input)

