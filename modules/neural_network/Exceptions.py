#!/usr/bin/env python
# coding: utf-8

# In[1]:


class ActivationException(Exception):
    def __init__(self, arg):
        self.strerror = 'ActivationException: ' + arg


# In[2]:


class ModelException(Exception):
    def __init__(self, arg):
        self.strerror = 'ModelException: ' + arg


# In[3]:


class LayerException(Exception):
    def __init__(self, arg):
        self.strerror = 'LayerException: ' + arg


# In[4]:


class ShapeException(Exception):
    def __init__(self, arg):
        self.strerror = 'ShapeException: ' + arg


# In[5]:


class LossException(Exception):
    def __init__(self, arg):
        self.strerror = 'LossException: ' + arg

