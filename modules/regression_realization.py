#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# ### Реализация линейной регрессии

# In[2]:


def stochastic_gradient_step(X, y, w, train_ind, eta=0.01):
    # считаем размер шага
    L = X.shape[0]
    # Считаем градиент по формуле
    grad = X[train_ind].dot(X[train_ind].dot(w) - y[train_ind])
    # Возвращаем новые значения весов
    return w - 2 * eta * grad / L


# Напишите функцию *linear_prediction*, которая принимает на вход матрицу *X* и вектор весов линейной модели *w*, а возвращает вектор прогнозов в виде линейной комбинации столбцов матрицы *X* с весами *w*.
# 
# $$\Large y = Xw$$

# In[3]:


def mserror(y, y_pred):
    return np.mean((y - y_pred)**2)


# In[4]:


def linear_prediction(X, w):
    return np.dot(X, w)


# Функция *stochastic_gradient_descent* реализует стохастический градиентный спуск для линейной регрессии. Функция принимает на вход следующие аргументы:**
# - X - матрица, соответствующая обучающей выборке
# - y - вектор значений целевого признака
# - w_init - вектор начальных весов модели
# - eta - шаг градиентного спуска (по умолчанию 0.01)
# - max_iter - максимальное число итераций градиентного спуска (по умолчанию 10000)
# - max_weight_dist - максимальное евклидово расстояние между векторами весов на соседних итерациях градиентного спуска,
# при котором алгоритм прекращает работу (по умолчанию 1e-8)
# - seed - число, используемое для воспроизводимости сгенерированных псевдослучайных чисел (по умолчанию 42)
# - verbose - флаг печати информации (например, для отладки, по умолчанию False)
# 
# На каждой итерации в вектор (список) должно записываться текущее значение среднеквадратичной ошибки. Функция должна возвращать вектор весов $w$, а также вектор (список) ошибок.

# In[5]:


def stochastic_gradient_descent(X, y, w_init, eta=1e-2, max_iter=1e4,
                                min_weight_dist=1e-8, seed=42, verbose=False):
    # Инициализируем расстояние между векторами весов на соседних
    # итерациях большим числом. 
    weight_dist = np.inf
    # Инициализируем вектор весов
    w = w_init
    # Сюда будем записывать ошибки на каждой итерации
    errors = []
    # Счетчик итераций
    iter_num = 0
    # Будем порождать псевдослучайные числа 
    # (номер объекта, который будет менять веса), а для воспроизводимости
    # этой последовательности псевдослучайных чисел используем seed.
    np.random.seed(seed)
    # Длина выборки
    L = X.shape[0]
    # Основной цикл
    while weight_dist > min_weight_dist and iter_num < max_iter:
        # порождаем псевдослучайный 
        # индекс объекта обучающей выборки
        random_ind = np.random.randint(X.shape[0])
        # Считаем следуюзий вектор весов
        w_next = stochastic_gradient_step(X, y, w, random_ind)
        # Считаем текущую ошибку
        cur_error = mserror(y, linear_prediction(X, w_next))
        # Пересчитываем расстояние между векторами
        weight_dist = np.sqrt(np.sum((w - w_next)**2))
        # Присваиваем новое значение вектора весов
        w = w_next
        # Добавляем ошибку на данной итерации в массив
        errors.append(cur_error)
        # Переходим на следующий шаг итерации
        iter_num += 1
        
    return w, errors


# In[6]:


class Regression_Linear:
    def __init__(self):
        pass
    
    def fit(self, X_train, y_train, n_iter=10**5):
        w_init = np.zeros(X_train.shape[1])
        self.w_, self.errors = stochastic_gradient_descent(X_train, y_train, w_init, max_iter=n_iter)
        pass
        
    def predict(self, X_test):
        return np.dot(X_test, self.w_)


# ### Реализация дерева решений

# Функция *tree_predict* возвращает среднее арифметическое по значениям целевого признака

# In[7]:


def tree_predict(X):
    return X[:, -1].mean()


# H - дисперсия ответов выборки
# 
# $\bar{y}(X)=\frac{1}{|X|}\sum_{i\in X}y_i$
# 
# $H(X)=\frac{1}{|X|}\sum_{i\in X}(y_i-\bar{y}(X))^2$

# In[8]:


def tree_H(X):
    if(X.shape[0] == 0):
        return np.inf
    # ответы y в последнем столбце
    y_ = (np.sum(X[-1])) / X.shape[0]
    return (np.sum(X[-1] - y_)) / X.shape[0]


# Функция *tree_Q* считает функционал ошибки, который минимизируется

# In[9]:


def tree_Q(X_m, j, t):
    X_l = X_m[np.where(X_m[:, j] <= t)]
    X_r = X_m[np.where(X_m[:, j] > t)]
    return (X_l.shape[0] * tree_H(X_l) + X_r.shape[0] * tree_H(X_r)) / X_m.shape[0]


# Функция *tree_select_condition* выбирает лучшие j, t на каждом шаге дерева

# In[10]:


def tree_select_condition(X_train):
    best_Q = np.inf
    j = 0
    t = 0
    # Убираем целевой признак
    for i in range(X_train.shape[1] - 1):
        X_i = X_train[:, i]
        unique_t = sorted(np.array(list(set(X_i))))
        for ind_t in range(len(unique_t) - 1):
            t_i = unique_t[ind_t]
            q = tree_Q(X_train, i, t_i)
            if q < best_Q:
                j = i
                t = t_i
    return j, t


# Функция *add_prediction* добавляет значения в словарь

# In[11]:


def add_prediction(X_train, X_test, y):
    pred_value = tree_predict(X_train)
    for x in X_test:
        y[x] = pred_value
    return y


# Функция *make_tree* отвечает за построение дерева и предсказание вершин. *idx* - индексы в массиве теста

# In[12]:


def make_tree(X_train,
              X_test,
              idx,
              max_depth, 
              min_samples_split,
              y={}, 
              n_iter=0):
    # Проверяем критерии останова
    # Если на тестовой выборке нет данных значений - возврат
    if X_test.shape[0] == 0:
        return y
    # Если глубина дерева достигла максимума
    if max_depth and n_iter >= max_depth:
        return add_prediction(X_train, idx, y)
    
    # Если число различных значений целевого признака
    num_samples = len(list(set(X_train[:, -1])))
    # Или меньше, чем минимально разрешенное для деления
    if num_samples < min_samples_split:
        return add_prediction(X_train, idx, y)
    # Ищем j, t
    j, t = tree_select_condition(X_train)
    # Разбивает тренировочную и тестовую выборку на 2 части
    X_train_j = X_train[:, j]
    X_test_j = X_test[:, j]
    X_l = X_train[np.where(X_train_j <= t)]
    X_r = X_train[np.where(X_train_j > t)]
    idx_l = np.where(X_test_j <= t)
    idx_r = np.where(X_test_j > t)
    X_test_l = X_test[idx_l]
    X_test_r = X_test[idx_r]
    y_new = make_tree(X_l, X_test_l, idx[idx_l], max_depth, min_samples_split, y, n_iter + 1)
    return make_tree(X_r, X_test_r, idx[idx_r], max_depth, min_samples_split, y_new, n_iter + 1)


# In[13]:


class Regression_Decision_Tree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        pass
    
    def fit(self, X_train, y_train):
        self.X = np.hstack((X_train, y_train.reshape((-1, 1))))
        pass
        
    def predict(self, X_test):
        idx = np.array([i for i in range(X_test.shape[0])])
        pred_dict = make_tree(self.X, X_test, idx, 
                              max_depth=self.max_depth, 
                              min_samples_split=self.min_samples_split)
        pred = np.array(list(map(lambda x: pred_dict[x], idx)))
        return pred


# ### Реализация Elastic Net

# $Q = 1 / (2 * n_{samples}) * ||y - Xw||^2_2
# + alpha * l1_{ratio} * ||w||_1
# + 0.5 * alpha * (1 - l1_{ratio}) * ||w||^2_2 \rightarrow min$

# $grad Q $

# In[14]:


def stochastic_gradient_step_EN(X, y, w, train_ind, eta=0.01, alpha=0, l1_ratio=0.5):
    # Шагаем по 1 объекту, следовательно, n_samples = 1
    # Считаем градиент по формуле
    grad_main = X[train_ind].dot(X[train_ind].dot(w) - y[train_ind])
    grad_w1 = np.array([1 if w_i > 0 else -1 if w_i < 0 else 0 for w_i in w])
    grad_w2 = w
    # Возвращаем новые значения весов
    return w - eta * grad_main + alpha * l1_ratio * grad_w1 + alpha * (1 - l1_ratio) * grad_w2


# Напишите функцию *linear_prediction*, которая принимает на вход матрицу *X* и вектор весов линейной модели *w*, а возвращает вектор прогнозов в виде линейной комбинации столбцов матрицы *X* с весами *w*.
# 
# $$\Large y = Xw$$

# Функция *stochastic_gradient_descent* реализует стохастический градиентный спуск для линейной регрессии. Функция принимает на вход следующие аргументы:**
# - X - матрица, соответствующая обучающей выборке
# - y - вектор значений целевого признака
# - w_init - вектор начальных весов модели
# - eta - шаг градиентного спуска (по умолчанию 0.01)
# - max_iter - максимальное число итераций градиентного спуска (по умолчанию 10000)
# - max_weight_dist - максимальное евклидово расстояние между векторами весов на соседних итерациях градиентного спуска,
# при котором алгоритм прекращает работу (по умолчанию 1e-8)
# - seed - число, используемое для воспроизводимости сгенерированных псевдослучайных чисел (по умолчанию 42)
# - verbose - флаг печати информации (например, для отладки, по умолчанию False)
# 
# На каждой итерации в вектор (список) должно записываться текущее значение среднеквадратичной ошибки. Функция должна возвращать вектор весов $w$, а также вектор (список) ошибок.

# In[15]:


def stochastic_gradient_descent_EN(X, y, w_init, eta=1e-2, max_iter=1e4,
                                min_weight_dist=1e-8, seed=42, verbose=False, 
                                   alpha=0, l1_ratio=0.5):
    # Инициализируем расстояние между векторами весов на соседних
    # итерациях большим числом. 
    weight_dist = np.inf
    # Инициализируем вектор весов
    w = w_init
    # Сюда будем записывать ошибки на каждой итерации
    errors = []
    # Счетчик итераций
    iter_num = 0
    # Будем порождать псевдослучайные числа 
    # (номер объекта, который будет менять веса), а для воспроизводимости
    # этой последовательности псевдослучайных чисел используем seed.
    np.random.seed(seed)
    # Длина выборки
    L = X.shape[0]
    # Основной цикл
    while weight_dist > min_weight_dist and iter_num < max_iter:
        # порождаем псевдослучайный 
        # индекс объекта обучающей выборки
        random_ind = np.random.randint(X.shape[0])
        # Считаем следуюзий вектор весов
        w_next = stochastic_gradient_step_EN(X, y, w, random_ind, 
                                             alpha=alpha, l1_ratio=l1_ratio)
        # Считаем текущую ошибку
        cur_error = mserror(y, linear_prediction(X, w_next))
        # Пересчитываем расстояние между векторами
        weight_dist = np.sqrt(np.sum((w - w_next)**2))
        # Присваиваем новое значение вектора весов
        w = w_next
        # Добавляем ошибку на данной итерации в массив
        errors.append(cur_error)
        # Переходим на следующий шаг итерации
        iter_num += 1
        
    return w, errors


# In[16]:


class Regression_EN:
    def __init__(self, alpha=0, l1_ratio=0.5):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        pass
    
    def fit(self, X_train, y_train, n_iter=10**5):
        w_init = np.zeros(X_train.shape[1])
        self.w_, self.errors = stochastic_gradient_descent_EN(X_train, y_train, w_init, 
                                                           max_iter=n_iter,
                                                           alpha=self.alpha,
                                                          l1_ratio=self.l1_ratio)
        pass
        
    def predict(self, X_test):
        return np.dot(X_test, self.w_)

