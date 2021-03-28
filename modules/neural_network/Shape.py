#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from Exceptions import ShapeException


# In[2]:


class Shape:
    '''
    Данный класс отвечает за инициализацию и работу с размерами слоя.
    Структура: (batch_size, dim_1, ..., dim_n)
    Минимум 2 размерности
    '''
    __slots__ = 'shape'

    def __init__(self, shape, *args, **kwargs):
        # Если передают аргументы кроме shape, возвращаем ошибку
        if args:
            raise ShapeException("Encountered unknown arguments: {}.".format(*args))
        if kwargs:
            raise ShapeException("Encountered unknown arguments: {}.".format(kwargs))

        if isinstance(shape, int):
            self.shape = (1, shape)
        elif isinstance(shape, (tuple, list)):
            for dim in shape:
                if type(dim) != int:
                    raise ShapeException("Non-integer dimensions are not supported: encountered {} in {}.".
                                         format(dim, shape))
            if len(shape) > 1:
                self.shape = shape
            elif len(shape) == 1:
                self.shape = (1, *shape)
            else:
                raise ShapeException("Negative or zero shape dimension encountered.")
        elif isinstance(shape, Shape):
            self.shape = shape.shape
        else:
            raise ShapeException("Can't cast shape {} to a tuple format.".format(shape))

    def reshape(self, new_shape):
        self.__init__(new_shape)

    def change_batch_size(self, new_batch_size):
        self.reshape((new_batch_size, *self.shape[1:]))

    def __getitem__(self, item):
        try:
            return self.shape[item]
        except IndexError:
            raise ShapeException("Can't slice the shape using index {}.".format(item))

    def __str__(self):
        return '(' + ', '.join([str(dim) if dim is not None else '?' for dim in self.shape]) + ')'

    def __format__(self, format_spec):
        return "{r:{f}}".format(r=self.__str__(), f=format_spec)

    def __eq__(self, other):
        if isinstance(other, tuple):
            return self.shape == other
        elif isinstance(other, list):
            return self.shape == tuple(other)
        elif isinstance(other, (Shape, np.ndarray)):
            return self.shape == other.shape
        else:
            return ShapeException("Can't compare {} to {}.".format(self, other))

    def __ne__(self, other):
        if isinstance(other, tuple):
            return self.shape != other
        elif isinstance(other, list):
            return self.shape != tuple(other)
        elif isinstance(other, (Shape, np.ndarray)):
            return self.shape != other.shape
        else:
            return ShapeException("Can't compare {} to {}.".format(self, other))

    def __call__(self, *args, **kwargs):
        return self.get()

