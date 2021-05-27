#!/usr/bin/env python
# coding: utf-8

# # Деплой в Heroku
# 
# https://test-ml-1.herokuapp.com/

# In[2]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import ElasticNet, Ridge, SGDRegressor
import pickle
import streamlit as st
import tensorflow
from tensorflow import keras


# In[3]:


from keras.layers import Dense, InputLayer
from keras.models import Sequential
from keras.activations import *
from keras.callbacks import Callback


# # Предобработка данных

# Проведем предобработку данных.
# 
# Разделим данные на обучение и тест, нормализуем их.

# In[3]:


@st.cache
def prepared_dataset(df, random_state=None):
    X = df.drop('price', axis=1)
    y = df.price.astype('int64')
    numeric_cols = np.array(['symboling', 'normalized_losses', 'wheel_base', 'length', 'width',
       'height', 'curb_weight', 'num_of_cylinders', 'engine_size', 'bore',
       'stroke', 'compression_ratio', 'horsepower', 'peak_rpm', 'city_mpg',
       'highway_mpg'])
    X_numeric = X.loc[:, numeric_cols]
    categorical_cols = list(set(X.columns.values.tolist()) - set(numeric_cols))
    X_categorical = X.loc[:, categorical_cols]
    for col in categorical_cols:
        X_categorical[col] = X_categorical[col].astype('string')
    # Разделим на обучение и тест
    X_train, X_test, X_train_cat, X_test_cat, X_train_num, X_test_num, y_train, y_test = train_test_split(X, X_categorical, X_numeric, y, test_size=0.15, random_state=random_state)
    # Номализуем значения
    scaler = StandardScaler()
    scaler.fit(X_train_num)

    X_train_num_sc = scaler.transform(X_train_num)
    X_test_num_sc = scaler.transform(X_test_num)
    # Соединяем значения воедино
    X_train_transform = np.hstack((X_train_num_sc, X_train_cat))
    X_test_transform = np.hstack((X_test_num_sc, X_test_cat))
    return X_train_transform.astype('float32'), y_train.astype('float32'), X_test_transform.astype('float32'), y_test.astype('float32')


# # Обучение алгоритмов

# ## Нейронная сеть

# In[4]:


def neural_network(parameters={'input_shape':69, 'num_neurons':3, 'num_layers':1, 'activations':['relu'],
                              'optimizer':'sgd', 'loss':'mse'}):
    input_shape = parameters['input_shape']
    num_neurons = parameters['num_neurons']
    num_layers = parameters['num_layers']
    activations = parameters['activations']
    optimizer = parameters['optimizer']
    loss = parameters['loss']
    
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    for i in range(num_layers):
        model.add(Dense(num_neurons, activation=activations[i]))
    model.add(Dense(1, activation='relu'))
    model.compile(optimizer, loss)
    return model


# ## Линейные алгоритмы

# Объявим доступные модели

# In[58]:


def init_model(model="", parameters={}):
    new_params = {}
    if model == "elastic_net":
        regressor = ElasticNet()
    elif model == "sgd_regressor":
        regressor = SGDRegressor()
    elif model == "ridge":
        regressor = Ridge()
    elif model == 'neural_network':
        return neural_network(parameters)
    else:
        regressor = ElasticNet()
    # get all available parameters
    available_params = set(regressor.get_params().keys()).intersection(set(parameters.keys()))
    params = {a_p:parameters[a_p] for a_p in available_params}
    regressor.set_params(**params)
    return regressor


# [Keras callback](https://keras.io/guides/writing_your_own_callbacks/)

# In[7]:


class CustomCallback(Callback):
    def __init__(self, epochs=1):
        super(CustomCallback, self).__init__()
        self.epochs = epochs
        self.pr_bar = st.progress(0)
        pass
    
    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        if epoch < self.epochs:
            progress = float(epoch) / self.epochs
        else:
            progress = 100
        self.pr_bar.progress(progress)
        pass


# Метод <code>trainee_model</code> обучает модель с гиперпараметрами

# In[6]:


def trainee_model(X_train, y_train, model="", parameters={'epochs':1}):
    regressor = init_model(model, parameters)
    if model == 'neural_network':
        regressor.fit(X_train, y_train, epochs=parameters['epochs'],
                     callbacks=[CustomCallback(parameters['epochs'])])
    else:
        regressor.fit(X_train, y_train)
    return regressor


# Метод <code>process_prediction</code> выдает метрики качества для модели

# In[30]:


def process_prediction(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    st.write('R^2: {}'.format(r2_score(y_test, y_pred)))
    st.write('MSE: {}'.format(mean_squared_error(y_test, y_pred)))
    st.write('RMSE: {}'.format(mean_squared_error(y_test, y_pred, squared=False)))
    st.write('MAE: {}'.format(mean_absolute_error(y_test, y_pred)))


# # Сериализация модели

# # Pickle [docs](https://docs.python.org/3/library/pickle.html)

# Сериализация в байтовое представление

# In[33]:


def serialize_model(model):
    return pickle.dumps(model)

def deserialize_model(bin_data):
    return pickle.loads(bin_data)


# Для сериализации/десериализации в/из файла можно использовать следующие функции

# In[34]:


def serialize_model_file(model, filename="ser.dat"):
    with open(filename, 'wb') as fin:
        pickle.dump(model, fin)
    return filename

def deserialize_model_file(filename="ser.dat"):
    with open(filename, 'rb') as fout:
        model = pickle.load(fout)
    return model


# # [STREAMLIT статья](https://www.notion.so/a53e9e35dc4f482f889e2a3f516be9fd) [docs](https://docs.streamlit.io/en/stable/)

# Загрузка данных из файла

# In[9]:


@st.cache
def load_description(filename='../data/automobile_description.md'):
    with open(filename, 'r', encoding="utf-8") as fin:
        descr_auto = fin.read()
        return descr_auto


# In[53]:


@st.cache
def load_data(filename='../data/automobile_preprocessed.csv'):
    df = pd.read_csv(filename, sep=';')
    return df


# In[12]:


def main():
    page = st.sidebar.selectbox('Choose a page', ['Description', 'Regressor'])
    df = load_data('../data/automobile_preprocessed.csv')
    if page == 'Description':
        st.header('Automobile dataset description')
        description_task = '''## Task 3
Подготовьте данные, уберите аномалии. Разбейте датасет, проведите обучение моделей: 
- Линейная регрессия;
- Регрессия дерева решений;
- LASSO;
- Гребневая регрессия;
- Elastic Net регрессия.

Найдите реализации методов в sklearn, оставьте в нотбуке ссылки на документацию. Найдите наилучшие гиперпараметры. Оцените качество моделей: R2, Mean Square Error(MSE), Root Mean Square Error(RMSE),  mean absolute error (MAE). Свои действия снабжайте пояснениями.'''
        st.write(description_task)
        st.dataframe(df)
        description = load_description()
        st.write(description)
    else:
        st.header('Automobile dataset')
        # Set model type
        model_type = st.selectbox('Select model', ['elastic_net', 'ridge', 'sgd_regressor', 'neural_network'])
        parameters = {}
        
        if model_type in ['elastic_net', 'ridge', 'sgd_regressor']:
            alpha = st.slider('Select param alpha', min_value=0.0, max_value=3.0, value=1.0, step=0.1, key='alpha')
            parameters['alpha'] = alpha
            
        if model_type == 'elastic_net':
            fit_intercept = st.selectbox('Select fit intercept', [True, False], key='is_fit_intercept')
            parameters['fit_intercept'] = fit_intercept
            max_iter = st.slider('Select param max_iter', min_value=1, max_value=50, value=1, step=1, key='max_iter')
            parameters['max_iter'] = max_iter
            positive = st.selectbox('Select is all params are positive', [True, False], key='is_positive')
            parameters['positive'] = positive
        elif model_type == 'ridge':
            normalize = st.selectbox('Select if normalize parameters', [True, False], key='is_normalize')
            parameters['normalize'] = normalize
        elif model_type == 'sgd_regressor':
            penalty = st.selectbox('Select penalty', ['l2', 'l1', 'elasticnet'], key='penalty')
            parameters['penalty'] = penalty
            if penalty == 'elasticnet':
                l1_ratio = st.slider('Select param l1_ratio', min_value=0.0, max_value=1.0, value=0.5, step=0.05, key='l1_ratio')
                parameters['l1_ratio'] = l1_ratio
        elif model_type == 'neural_network':
            num_layers = st.slider('Select number of layers', min_value=1, max_value=3, step=1, key='num_layers')
            parameters['num_layers'] = num_layers
            activations = []
            for i in range(num_layers):
                activation = st.selectbox('Select activation for {} layer'.format(i + 1), 
                                          ['relu', 'sigmoid', 'softmax', 'softsign', 'tanh', 'selu', 'elu', 'exponential'], 
                                          key='activation_{}'.format(i))
                activations.append(activation)
            parameters['activations'] = activations
            num_neurons = st.slider('Select number of neurons on each hidden layer', min_value=1, max_value=10, step=1, key='num_neurons')
            parameters['num_neurons'] = num_neurons
            optimizer = st.selectbox('Select optimizer function', ['SGD', 'RMSprop', 'Adam', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl'], key='optimizer')
            parameters['optimizer'] = optimizer
            loss = st.selectbox('Select loss function', ['mse', 'mae'], key='loss')
            parameters['loss'] = loss
            epochs = st.slider('Select number of epochs', min_value=1, max_value=2000, value=1, step=6, key='epochs')
            parameters['epochs'] = epochs

        X_train, y_train, X_test, y_test = prepared_dataset(df)
        if model_type == 'neural_network':
            input_shape = X_train.shape[1]
            parameters['input_shape'] = input_shape
        model = trainee_model(X_train, y_train, model=model_type, parameters=parameters)
        if model_type != 'neural_network':
            # Сериализация модели
            model_bytes = serialize_model(model)
            # Десериализация модели
            model_deser = deserialize_model(model_bytes)
        process_prediction(model, X_test, y_test)
    st.write("By Konstantin Ushakov, 2021.")


# In[13]:


if __name__ == "__main__":
    main()

