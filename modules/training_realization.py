#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# ### Реализация KNN

# In[2]:


def dist_2(x1, x2):
    return np.sum((x1 - x2)**2)**0.5


# In[3]:


class Classifier_KNN:
    def __init__(self):
        pass
    
    def fit(self, X_train, y_train):
        # Сохраняем созданные значения
        self.X_train = X_train
        self.y_train = y_train
        pass
    
    def predict(self, X_test, n_neighbors=5):
        # Создаем массив предсказанных значений
        y_pred = np.array([])
        size_train = len(self.X_train)
        for x in X_test:
            # Массив пар (расстояние, y).
            dists = [(dist_2(x, self.X_train[i]), self.y_train[i]) for i in range(size_train)]
            # Сортируем по расстоянию. 
            dists.sort(key=lambda x: x[0])
            if n_neighbors > self.X_train.shape[0]:
                n_neighbors = self.X_train.shape[0]
            # Берем первые n_neighbors соседей - от них отщепляем только y
            neibours = [n[1] for n in dists[:n_neighbors]]
            y_dict = {}
            max_val = 0
            y_cur = neibours[0]
            # Подсчет каких значений y получается больше
            for elem in neibours:
                prev_val = y_dict.setdefault(elem, 0)
                y_dict[elem] = prev_val + 1
                if max_val <= prev_val:
                    max_val = prev_val + 1
                    y_cur = elem
            # Добавление подсчитанного значения в массив
            y_pred = np.append(y_pred, y_cur)
        return y_pred


# ### Реализация Naive Bayes

# $a(x) = argmax_y P(x|y)P(y)$
# 
# $P(x|y) = P(x_1 | y)P(x_2 | y)...P(x_N | y)$, где $x_k$ - $k$-ый признак объекта $x$
# 
# $P(y) = \frac{l_y}{l}$
# 
# Бинарные признаки:
# 
# $P(x_k | y) = \frac{1}{l_y} доля(x_k, y)$
# 
# Числовые признаки:
# 
# Восстановить распределение (более сложные методы)

# Ограничимся рассмотрением только бинарных признаков (либо любых числовых, с разделителем 0.5)

# In[4]:


class Classifier_NB_bin:
    def __init__(self):
        pass
    
    # Для бинарных значений класса
    def fit(self, X_train, y_train):
        self.classes = np.array(list(set(y_train)))
        self.n_class = self.classes.shape[0]
        # Сохраняем созданные значения
        self.X_train = X_train
        self.y_train = y_train
        self.p = np.array([])
        self.l_y = np.array([])
        # Считаем вероятности каждого класса
        for i in range(self.n_class):
            self.l_y = np.append(self.l_y, np.sum([1 if y == self.classes[i] else 0 for y in y_train]))
            self.p = np.append(self.p, self.l_y[i] / X_train.shape[0])
        # каждая пара - это доля объектов, принадлежащих к классу (x,y) по тестовой выборке:
        # (x_i, y) = [[(y_0, 0), (y_0, 1)], ..., [(y_i, 0), (y_i, 1)], ..., [(y_n, 0), (y_n, n)]]
        # Инициализируем массив таких кортежей
        count_p_x_0 = np.array([np.array([0 for j in range(self.n_class)]) for i in range(X_train.shape[1])])
        count_p_x_1 = np.array([np.array([0 for j in range(self.n_class)]) for i in range(X_train.shape[1])])
        
        for x, y in list(zip(X_train, y_train)):
            j, = np.where(y == self.classes)
            # Для каждого признака x_i заносим в класс значение
            for i, x_i in enumerate(x):
                # x - бинарный признак, принимает значения 0 и 1
                if x_i > 0.5:
                    # y, 1
                    count_p_x_1[i][j[0]] += 1
                else:
                    # y, 0
                    count_p_x_0[i][j[0]] += 1
        # Считаем долю признаков каждого типа - делим на число записей в обучении (l_y)
        
        self.p_x_0 = count_p_x_0 / (np.array([self.l_y for x in range(X_train.shape[1])]))
        self.p_x_1 = count_p_x_1 / (np.array([self.l_y for x in range(X_train.shape[1])]))
        print(self.l_y)
        pass
    
    def predict(self, X_test):
        # Создаем массив предсказанных значений
        y_pred = np.array([])
        for x in X_test:
            # Вычисляем вероятность принадлежать к каждому классу
            # Инициализируем вероятностями принадлежности i-му классу
            p_x = np.array(self.p, copy=True)
            for i, x_i in enumerate(x):
                if x_i > 0.5:
                    p_x *= self.p_x_1[i]
                else:
                    p_x *= self.p_x_0[i]
            #Находим индекс самого вероятного класса
            j, = np.where(p_x == np.max(p_x))
            y_cur = self.classes[j[0]]
            y_pred = np.append(y_pred, y_cur)
        return y_pred


# ### Реализация Support Vector Machines

# Hingeloss: $H = max(0, 1 - y (w^T x))$
# 
# Зазор margin: $m = y (w^T x)$
# 
# Soft-margin SVM (Q): $max(0, 1 - y w^T x) + \alpha (w^T w) / 2$
# 
# $\eta$ - eta

# Авторская реализация (https://habr.com/ru/company/ods/blog/484148/) (soft-margin)

# In[5]:


def add_bias_feature(a):
    a_extended = np.zeros((a.shape[0],a.shape[1]+1))
    a_extended[:,:-1] = a
    a_extended[:,-1] = int(1)  
    return a_extended


class Classifier_SVM_realized(object):
    
    __class__ = "CustomSVM"
    __doc__ = """
    This is an implementation of the SVM classification algorithm
    Note that it works only for binary classification
    
    #############################################################
    ######################   PARAMETERS    ######################
    #############################################################
    
    etha: float(default - 0.01)
        Learning rate, gradient step
        
    alpha: float, (default - 1.0)
        Regularization parameter in 0.5*alpha*||w||^2
        
    epochs: int, (default - 200)
        Number of epochs of training
      
    #############################################################
    #############################################################
    #############################################################
    """
    
    def __init__(self, etha=0.01, alpha=0.1, epochs=200):
        self._epochs = epochs
        self._etha = etha
        self._alpha = alpha
        self._w = None
    

    def fit(self, X_train, Y_train, verbose=False): #arrays: X; Y =-1,1
        
        if len(set(Y_train)) != 2:
            raise ValueError("Number of classes in Y is not equal 2!")
        
        X_train = add_bias_feature(X_train)
        self._w = np.random.normal(loc=0, scale=0.05, size=X_train.shape[1])
        
        for epoch in range(self._epochs):
            for i,x in enumerate(X_train):
                margin = Y_train[i]*np.dot(self._w,X_train[i])
                if margin >= 1: # классифицируем верно
                    self._w = self._w - self._etha*self._alpha*self._w/self._epochs
                else: # классифицируем неверно или попадаем на полосу разделения при 0<m<1
                    self._w = self._w + self._etha*(Y_train[i]*X_train[i] - self._alpha*self._w/self._epochs)
        pass

    def predict(self, X:np.array) -> np.array:
        y_pred = []
        X_extended = add_bias_feature(X)
        for i in range(len(X_extended)):
            sign = np.sign(np.dot(self._w,X_extended[i]))
            if sign > 0.5:
                y_pred = np.append(y_pred, 1)
            else:
                y_pred = np.append(y_pred, 0)
            
        return np.array(y_pred)
    


# Реализация (soft_margin)

# In[6]:


def add_ones(x):
    ones = np.ones(x.shape[0])
    ones = np.reshape(ones, (ones.shape[0], 1))
    return np.hstack((x, ones))
        
class Classifier_SVM:
    def __init__(self, eta=1e-2, alpha=1e-1, epochs=200):
        self.eta = eta
        self.alpha = alpha
        self.epochs = epochs
        pass
    
    def fit(self, X_train, y_train):
        X_train = X_train.astype(np.int)
        # Сохраняем созданные значения
        self.X_train = add_ones(X_train).astype(np.int)
        self.y_train = y_train.astype(np.int)
        # Начинаем с w = [0, 0, ..., 0]
        # self._w = np.zeros(self.X_train.shape[1])
        self._w = np.random.normal(loc=0, scale=0.05, size=self.X_train.shape[1])
        # C = eta * alpha / epochs
        C = self.eta * self.alpha / self.epochs
        for epoch in range(self.epochs):
            for j, x in enumerate(self.X_train):
                # w = w - eta * grad(Q)
                margin_cur = self.y_train[j] * np.dot(self._w, x)
                if margin_cur >= 1:
                    # ywx >= 1 => hinde_loss = 0 - правильная классификация
                    # grad(Q) = alpha * w
                    self._w += - C * self._w
                else:
                    # ywx < 1 => hinde_loss > 0 - неправильная классификация
                    # grad(Q) = alpha * w - yx
                    self._w += - C * self._w + self.eta * self.y_train[j] * x
        pass
    
    def predict(self, X_test):
        # Создаем массив предсказанных значений
        y_pred = np.array([])
        x_ext = add_ones(X_test)
        # Предсказываем как линейную модель y = wx
        # Так как только для бинарной классификации, берем знак
        for x in x_ext:
            sign = np.sign(np.dot(self._w, x))
            if sign > 0.5:
                y_pred = np.append(y_pred, 1)
            else:
                y_pred = np.append(y_pred, 0)
        return y_pred
    
    def hinde_loss(self, x, y):
        return np.max(0, 1 - self.margin(x, y))
    
    def margin(self, x, y):
        return y * np.dot(x, self._w)
    

