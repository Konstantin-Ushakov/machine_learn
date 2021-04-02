#!/usr/bin/env python
# coding: utf-8

# Ограничим максимальные значения, получаемые в экспоненте, 1e7

# In[ ]:


import numpy as np
from Exceptions import ActivationException


# In[2]:


def H(z, eps=1e-12):
    return np.where(z > -eps, 1, 0)

def sigmoid(z, a=1, eps=1e-12):
    ex = np.exp(-a * z)
    divisor = 1. + np.where(ex > 1e7, 1e7, ex)
    divisor = np.where(divisor == 0, divisor + eps, divisor)
    return 1. / divisor


# In[3]:


class Activation:
    '''
    Родительский класс активации, от него наследуется производные классы активации
    По умолчанию реализует тождественную функцию
    f(z) = z
    f'(z) = 1 (размерности, равной входящей)
    '''
    def __init__(self):
        pass
    
    def f(self, z, *args, **kwargs):
        return z
    
    def grad(self, z, *args, **kwargs):
        return np.ones_like(z)
    
    '''
    Callable object - можно вызвать как Object()
    '''
    def __call__(self, z, *args, **kwargs):
        return self.f(z, *args, **kwargs)


# In[4]:


class Step(Activation):
    '''
    Ступенька
    f(z) = 1 if x >= 0
    f'(z) = 0
    '''
    def __init__(self, eps=1e-12):
        self.eps = eps
        pass
    
    def f(self, z, *args, **kwargs):
        return H(z, eps=self.eps)
    
    def grad(self, z, *args, **kwargs):
        return np.zeros_like(z)


# In[5]:


class Sigmoid(Activation):
    '''
    f(z) = 1/(1+exp(-a*z))
    f'(z) = a*f(z) * (1-a*f(z))
    '''
    
    def __init__(self, a=1, eps=1e-12):
        self.a = a
        self.eps = eps
        pass
    
    def f(self, z, *args, **kwargs):
        return sigmoid(z, a=self.a, eps=self.eps)
    
    def grad(self, z, *args, **kwargs):
        f_z = self.f(z)
        return self.a * f_z * (1 - self.a * f_z)


# In[6]:


class HypTangens(Activation):
    '''
    params: a
    f(z) = th z/a = (e^(z/a) - e^(-z/a)) / (e^(z/a) + e^(-z/a))
    f'(z) = (1 + f(z))(1 - f(z))
    Параметр eps - константа для численной стабильности (деления на нуль)
    a != 0
    '''
    def __init__(self, a=1, eps=1e-12):
        self.a = a
        if self.a == 0:
            raise ActivationError('Param a on {} class can\'t be equal of zero'.format(self.__class__))
        self.eps = eps
        pass
    
    def f(self, z, *args, **kwargs):
        ex_z = np.exp(z / self.a)
        ex_z = np.where(ex_z > 1e7, 1e7, ex_z)
        ex_minus_z = np.exp(-z / self.a)
        ex_minus_z = np.where(ex_minus_z > 1e7, 1e7, ex_minus_z)
        divisor = ex_z + ex_minus_z
        divisor = np.where(divisor == 0, divisor + self.eps, divisor)
        return (ex_z - ex_minus_z) / divisor
    
    def grad(self, z, *args, **kwargs):
        f_z = self.f(z)
        return (1 + f_z) * (1 - f_z)


# In[7]:


class ArcTg(Activation):
    '''
    Arctangens
    f(z) = tg^-1(z)
    f'(z)=1/(z^2+1)
    Параметр eps - константа для численной стабильности (деления на нуль)
    '''
    
    def __init__(self, eps=1e-12):
        self.eps = eps
        pass
    
    def f(self, z, *args, **kwargs):
        return np.arctan(z)
    
    def grad(self, z, *args, **kwargs):
        divisor = z*z + 1.
        divisor = np.where(divisor == 0, divisor + self.eps, divisor)
        return 1. / divisor


# In[8]:


class Softsign(Activation):
    '''
    Softsign
    f(z) = z / (1 + |z|)
    f'(z) = 1 / (1 + |z|)^2
    Параметр eps - константа для численной стабильности (деления на нуль)
    '''
    
    def __init__(self, eps=1e-12):
        self.eps = eps
        pass
    
    def f(self, z, *args, **kwargs):
        divisor = 1. + np.abs(z)
        divisor = np.where(divisor == 0, divisor + self.eps, divisor)
        return z / divisor
    
    def grad(self, z, *args, **kwargs):
        denom = 1. + np.abs(z)
        divisor = denom * denom
        divisor = np.where(divisor == 0, divisor + self.eps, divisor)
        return 1. / divisor


# In[9]:


class ISRU(Activation):
    '''
    Обратный квадратный корень - Inversed Square Root Unit
    f(z) = z / (1 + a * z^2)^(1/2)
    f'(z) = (1 / (1 + a * z^2)^(1/2))^3
    Параметр eps - константа для численной стабильности (деления на нуль)
    '''
    
    def __init__(self, a=1, eps=1e-12):
        self.a = a
        self.eps = eps
        pass
    
    def f(self, z, *args, **kwargs):
        divisor = np.sqrt(1. + self.a*z*z)
        divisor = np.where(divisor == 0, divisor + self.eps, divisor)
        return z / divisor
    
    def grad(self, z, *args, **kwargs):
        divisor = np.sqrt(1. + self.a*z*z)
        divisor = np.where(divisor == 0, divisor + self.eps, divisor)
        return np.power(1. / divisor, 3)


# In[10]:


class ReLU(Activation):
    '''
        Rectified Linear Unit.
        f(z) = z (z >= 0), либо 0 (z<0).
        f'(z) = 1 (z>=0) либо 0 (z<0).
        Параметр eps - константа для численной стабильности (сравнения с нулем)
    '''
    
    def __init__(self, eps=1e-12):
        self.eps = eps
        pass
    
    def f(self, z, *args, **kwargs):
        return np.where(z < -self.eps, 0, z)
    
    def grad(self, z, *args, **kwargs):
        return np.where(z < -self.eps, 0, 1)


# In[11]:


class LeakyReLU(Activation):
    '''
        Rectified Linear Unit.
        f(z) = z (z >= 0), либо a*z (z<0).
        f'(z) = 1 (z>=0) либо a (z<0).
        Параметр eps - константа для численной стабильности (сравнения с нулем)
    '''
    
    def __init__(self, a=0.01, eps=1e-12):
        self.eps = eps
        self.a = a
        pass
    
    def f(self, z, *args, **kwargs):
        return np.where(z < self.eps, self.a * z, z)
    
    def grad(self, z, *args, **kwargs):
        return np.where(z < self.eps, self.a, 1)


# In[12]:


class ELU(Activation):
    '''
    Экспотенциальная линейная функция
    Exponential linear unit
    f(z) = a(e^z - 1) (z < 0) или z (z >= 0)
    f'(z) = f(z) + a (z < 0) или 1, (z >= 0)
    Параметр eps - константа для численной стабильности (сравнения с нулем)
    '''
    
    def __init__(self, a=1, eps=1e-12):
        self.eps = eps
        self.a = a
        pass
    
    def f(self, z, *args, **kwargs):
        ez = np.exp(z)
        ez = np.where(ez > 1e7, 1e7, ez)
        return np.where(z < self.eps, self.a * (ez - 1.), z)
    
    def grad(self, z, *args, **kwargs):
        return np.where(z < self.eps, self.f(z) + self.a, 1)


# In[13]:


class SELU(Activation):
    '''
    Масштабированная экспотенциальная линейная функция
    Scaled Exponential linear unit
    f(z) = lambda(a(e^z - 1) (z < 0) или z (z >= 0))
    f'(z) = lambda(f(z) + a (z < 0) или 1, (z >= 0))
    Параметр eps - константа для численной стабильности (сравнения с нулем)
    '''
    
    def __init__(self, a=1.67326, lambd=1.0507, eps=1e-12):
        self.eps = eps
        self.lambd = lambd
        self.a = a
        pass
    
    def f(self, z, *args, **kwargs):
        ez = np.exp(z)
        ez = np.where(ez > 1e7, 1e7, ez)
        return self.lambd * np.where(z < self.eps, self.a * (ez - 1.), z)
    
    def grad(self, z, *args, **kwargs):
        ez = np.exp(z)
        ez = np.where(ez > 1e7, 1e7, ez)
        return self.lambd * np.where(z < self.eps, self.a * ez, 1)


# In[14]:


class SReLU(Activation):
    '''
    Линейный S-выпрямитель
    S-shaped rectified linear activation unit
    f(z) =
        { t_l + a_l(z - t_l), z <= t_l
        { z, t_l < z < t_r
        { t_r + a_r(z - t_r), z >= t_r
    f'(z) = 
        { a_l, z <= t_l
        { 1, t_l < z < t_r
        { a_r, z >= t_r
    Параметр eps - константа для численной стабильности (сравнения с нулем)
    '''
    
    def __init__(self, t_l=-1, a_l=1, t_r=1, a_r=1, eps=1e-12):
        if t_r - t_l < eps:
            raise ActivationException('t_l is left border, must be less then t_r, got t_l {} and t_r {}'.
                                     format(t_l, t_r))
        self.t_l = t_l
        self.a_l = a_l
        self.t_r = t_r
        self.a_r = a_r
        self.eps = eps
        pass
    
    def f(self, z, *args, **kwargs):
        return np.array([[
            self.t_l + self.a_l * (x - self.t_l) if self.t_l - x < self.eps else
            self.t_r + self.a_r * (x - self.t_r) if x - self.t_r < self.eps else
            x for x in z_i] for z_i in z
        ])
    
    def grad(self, z, *args, **kwargs):
        return np.array([[
            self.a_l if self.t_l - x < self.eps else
            self.a_r if x - self.t_r < self.eps else
            1 for x in z_i] for z_i in z
        ])


# In[15]:


class ISRLU(Activation):
    '''
    Обратный квадратный линейный корень
    Inverse square root linear unit
    f(z) = z / (1 + a * z^2)^(1/2) (z < 0), или z (z >= 0)
    f'(z) = (1 / (1 + a * z^2)^(1/2))^3, или z (z >= 0)
    Параметр eps - константа для численной стабильности (сравнения с нулем, деление на нуль)
    '''
    
    def __init__(self, a=1, eps=1e-12):
        self.eps = eps
        self.a = a
        pass
    
    def f(self, z, *args, **kwargs):
        divisor = np.sqrt(1. + self.a*z*z)
        divisor = np.where(divisor == 0, divisor + self.eps, divisor)
        return np.where(z < self.eps, z / divisor, z)
    
    def grad(self, z, *args, **kwargs):
        divisor = np.sqrt(1. + self.a*z*z)
        divisor = np.where(divisor == 0, divisor + self.eps, divisor)
        return np.where(z < self.eps, np.power(1. / divisor, 3), z)


# In[16]:


class APL(Activation):
    '''
    Адаптивная кусочно-линейная функция
    Adaptive pievewise linear
    f(z) = max(0, z) + summ_s=1,S(a^s_i * max(0, -z + b^s_i))
    f'(z) = H(z) - summ_s=1,S(a^s_i * H(-z + b^s_i))
    где H(z) - ступенчатая функция Хевисайда (ступенька)
    Параметр eps - константа для численной стабильности (сравнения с нулем)
    '''
    
    def __init__(self, a, b, eps=1e-12):
        self.eps = eps
        self.a = a
        self.b = b
        if len(a) != len(b):
            raise ActivationException('lens are not equal: len(a) = {}, len(b) = {}'.
                                     format(len(a), len(b)))
        pass
    
    def f(self, z, *args, **kwargs):
        return np.max(np.zeros_like(z), z) + np.array([
            np.sum(a * np.max(0, - x + b)) for a, b, x in zip(self.a, self.b, self.x)
        ])
    
    def grad(self, z, *args, **kwargs):
        return self.H(z, eps=self.eps) - np.array([
            np.sum(a * self.H(- x + b, eps=self.eps)) for a, b, x in zip(self.a, self.b, self.x)
        ])


# In[17]:


class SoftPlus(Activation):
    '''
    SoftPlus
    f(z) = ln(1 + exp(z))
    f'(z) = 1 / (1 + exp(-z))
    Параметр eps - константа для численной стабильности (деления на нуль)
    '''
    
    def __init__(self, eps=1e-12):
        self.eps = eps
        pass
    
    def f(self, z, *args, **kwargs):
        ez = np.exp(z)
        ez = np.where(ez > 1e7, 1e7, ez)
        return np.log(1 + ez)
    
    def grad(self, z, *args, **kwargs):
        divisor = 1. + np.exp(-z)
        divisor = np.where(divisor == 0, divisor + self.eps, divisor)
        return 1. / divisor


# In[18]:


class BI(Activation):
    '''
    Вытянутая тождественная функция
    Bent Identity
    f(z) = (sqrt(z^2 + 1) - 1) / 2 + z
    f'(z) = z / (2 * sqrt(z^2 + 1)) + 1
    Параметр eps - константа для численной стабильности (деления на нуль)
    '''
    def __init__(self, eps=1e-12):
        self.eps = eps
        pass
    
    def f(self, z, *args, **kwargs):
        return (np.sqrt(z * z + 1.) - 1.) / 2. + z
    
    def grad(self, z, *args, **kwargs):
        divisor = 2. * np.sqrt(z * z + 1.)
        divisor = np.where(divisor == 0, divisor + self.eps, divisor)
        return z / divisor + 1.


# In[19]:


class SiLU(Activation):
    '''
    Сигмоидно-взвешенная линейная функция
    Sigmoid-weighted linear unit
    f(z) = z * sigmoid(z)
    f'(z) = f(z) + sigmoid(z) * (1 - f(z))
    Параметр eps - константа для численной стабильности (деления на нуль)
    '''
    def __init__(self, eps=1e-12):
        self.eps = eps
        pass
    
    def f(self, z, *args, **kwargs):
        return z * sigmoid(z, eps=self.eps)
    
    def grad(self, z, *args, **kwargs):
        f = self.f(z)
        return f + sigmoid(z, eps=self.eps) * (1 - f)


# In[20]:


class SoftExponential(Activation):
    '''
    SoftExponential
    f(z) = 
        { - (ln(1 - a(z + a))) / a, a < 0
        { z, a = 0
        { (exp(a * z) - 1) / a + a, a > 0
    f'(z) = 1 / (1 - a * (a + z)) (a < 0), или exp(a * z) (a >= 0)
    Параметр eps - константа для численной стабильности (сравнения с нулем, деление на нуль)
    '''
    def __init__(self, a=0, exp=1e-12):
        self.a = a
        self.eps = exp
        pass
    
    def f(self, z, *args, **kwargs):
        if self.a > self.eps:
            return - (np.log(1. - self.a * (z + self.a))) / self.a
        elif self.a < -self.eps:
            ez = np.exp(self.a * z)
            ez = np.where(ez > 1e7, 1e7, ez)
            return (ez - 1) / self.a + self.a
        return z
    
    def grad(self, z, *args, **kwargs):
        if self.a < -self.eps:
            divisor = 1 - self.a * (self.a + z)
            divisor = np.where(divisor == 0, divisor + self.eps, divisor)
            return 1. / divisor
        return np.exp(self.a * z)


# In[21]:


class Sin(Activation):
    '''
    Синусоида
    f(z) = sin(z)
    f'(z) = cos(z)
    '''
    def __init__(self):
        pass
    
    def f(self, z, *args, **kwargs):
        return np.sin(z)
    
    def grad(self, z, *args, **kwargs):
        return np.cos(z)


# In[22]:


class Sinc(Activation):
    '''
    Sinc
    f(z) = sin(z) / z (z != 0), или 1 (z = 0)
    f'(z) = cos(z) / z - sin(z) / z^2 (z != 0), или 0 (z = 0)
    Параметр eps - константа для численной стабильности (деления на нуль)
    '''
    def __init__(self, eps=1e-12):
        self.eps = eps
        pass
    
    def f(self, z, *args, **kwargs):
        divisor = z
        divisor = np.where(divisor == 0, divisor + self.eps, divisor)
        return np.where(z != 0, np.sin(z) / divisor, 1)
    
    def grad(self, z, *args, **kwargs):
        divisor = z
        divisor = np.where(divisor == 0, divisor + self.eps, divisor)
        return np.where(z != 0, np.sin(z) / divisor, 0)


# In[23]:


class Gauss(Activation):
    '''
    Гауссова
    f(z) = exp(-z^2)
    f'(z) = -2 * z * exp(-z^2)
    '''
    def __init__(self):
        pass
    
    def f(self, z, *args, **kwargs):
        ez = np.exp(-z * z)
        ez = np.where(ez > 1e7, 1e7, ez)
        return ez
    
    def grad(self, z, *args, **kwargs):
        return -2. * z * np.exp(-z * z)


# In[24]:


class Softmax(Activation):
    '''
        Softmax. Преобразует вектор к распределению вероятностей.
        f(z_i) = exp(z_i)/sum_k exp(z_k) = p_i.
        f'(z_i)_j = d/dz_j exp(z_i) / sum_k (exp(z_k)) - exp(z_i) * 1/(sum_k (exp(z_k))^2 * sum_k d/dz_j exp(z_k)
            i = j:
                f'(z_i)_i = exp(z_k) / sum_k (exp(z_k)) - [exp(z_i)/sum_k (exp(z_k))]^2 = p_i - p_i^2 = p_i (1-p_i)
            i =/= j:
                f'(z_i)_j = 0 - exp(z_i)*exp(z_j) / (sum_k exp(z_k))^2 =
                 = -[exp(z_i)/sum_k exp(z_k)]*[exp(z_j)/sum_k exp(z_k)] = - p_i * p_j
        Параметр eps - константа для численной стабильности (деления на нуль)
    '''
    
    def __init__(self, eps=1e-12):
        self.eps = eps
        pass
    
    def f(self, z, *args, **kwargs):
        ez = np.exp(z)
        ez = np.where(ez > 1e7, 1e7, ez)
        divisor = np.sum(ez)
        divisor = np.where(divisor == 0, divisor + self.eps, divisor)
        return ez / divisor
    
    def grad(self, z, *args, **kwargs):
        J = np.zeros((z.shape[0], z.shape[1], z.shape[1]))
        p = self.f(z)
        for b in range(J.shape[0]):
            for i in range(J.shape[1]):
                for j in range(J.shape[2]):
                    J[b, i, j] = p[b, i] * (1 - p[b, i]) if i == j else -p[b, i] * p[b, j]
        return J


# In[25]:


class CustomActivation(Activation):
    __slots__ = '__f', '__grad', '__kwargs'
    
    def __init__(self, f, grad, **kwargs):
        self.__f = f
        self.__grad = grad
        self.__kwargs = kwargs
        pass
    
    def f(self, z, *args, **kwargs):
        try:
            return self.__f(z, *args, **self.__kwargs)
        except Exception:
            raise ActivationException('Custom activation function raised an error. Check funtion signature of {} to be equal to (z, *args, **kwargs).'.
                                      format(self.__f))
        pass
        
    def grad(self, z, *args, **kwargs):    
        try:
            return self.__grad(z, *args, **self.__kwargs)
        except Exception:
            raise ActivationException('Custom activation gradient raised an error. Check funtion signature of {} to be equal to (z, *args, **kwargs).'.
                                      format(self.__f))
        pass


# In[26]:


activation_alias = {
    'step': Step(),
    'sigmoid': Sigmoid(),
    'arc_tg': ArcTg(),
    'hyp_tg': HypTangens(),
    'softsign': Softsign(),
    'isru': ISRU(),
    'relu': ReLU(),
    'leaky_relu': LeakyReLU(),
    'elu': ELU(),
    'selu': SELU(),
    'srelu': SReLU(),
    'isrlu': ISRLU(),
    'softplus': SoftPlus(),
    'bent_identity': BI(),
    'silu': SiLU(),
    'softexponential': SoftExponential(),
    'sin': Sin(),
    'sinc': Sinc(),
    'gauss': Gauss(),
    'softmax': Softmax(),
}

# APL is not here - add by hands


# In[27]:


def register_activation(alias, activation):
    if issubclass(type(activation), Activation):
        raise ActivationException('Custom activation must inherit from Activation class')
    activation_alias[alias] = activation

