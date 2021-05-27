#!/usr/bin/env python
# coding: utf-8

# In[6]:


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


# In[7]:


from keras.layers import Dense, InputLayer
from keras.models import Sequential
from keras.activations import *


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


# Метод <code>trainee_model</code> обучает модель с гиперпараметрами

# In[29]:


def trainee_model(X_train, y_train, model="", parameters={'epochs':1}):
    regressor = init_model(model, parameters)
    if model == 'neural_network':
        regressor.fit(X_train, y_train, epochs=parameters['epochs'])
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

# In[53]:


@st.cache
def load_data(filename='../data/automobile_preprocessed.csv'):
    df = pd.read_csv(filename, sep=';')
    return df


# In[12]:


def main():
    page = st.sidebar.selectbox('Choose a page', ['Description', 'Regressor'])
    
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
        description = '''
        ## Автомобили [Источник](https://archive.ics.uci.edu/ml/datasets/Automobile)

### Описание данных

     Attribute:                Attribute Range:
     ------------------        -----------------------------------------------
  1. symboling:                -3, -2, -1, 0, 1, 2, 3.
  2. normalized-losses:        continuous from 65 to 256.
  3. make:                     alfa-romero, audi, bmw, chevrolet, dodge, honda,
                               isuzu, jaguar, mazda, mercedes-benz, mercury,
                               mitsubishi, nissan, peugot, plymouth, porsche,
                               renault, saab, subaru, toyota, volkswagen, volvo
  4. fuel-type:                diesel, gas.
  5. aspiration:               std, turbo.
  6. num-of-doors:             four, two.
  7. body-style:               hardtop, wagon, sedan, hatchback, convertible.
  8. drive-wheels:             4wd, fwd, rwd.
  9. engine-location:          front, rear.
 10. wheel-base:               continuous from 86.6 120.9.
 11. length:                   continuous from 141.1 to 208.1.
 12. width:                    continuous from 60.3 to 72.3.
 13. height:                   continuous from 47.8 to 59.8.
 14. curb-weight:              continuous from 1488 to 4066.
 15. engine-type:              dohc, dohcv, l, ohc, ohcf, ohcv, rotor.
 16. num-of-cylinders:         eight, five, four, six, three, twelve, two.
 17. engine-size:              continuous from 61 to 326.
 18. fuel-system:              1bbl, 2bbl, 4bbl, idi, mfi, mpfi, spdi, spfi.
 19. bore:                     continuous from 2.54 to 3.94.
 20. stroke:                   continuous from 2.07 to 4.17.
 21. compression-ratio:        continuous from 7 to 23.
 22. horsepower:               continuous from 48 to 288.
 23. peak-rpm:                 continuous from 4150 to 6600.
 24. city-mpg:                 continuous from 13 to 49.
 25. highway-mpg:              continuous from 16 to 54.
 26. price:                    continuous from 5118 to 45400.

8. Missing Attribute Values: (denoted by "?")
   Attribute #:   Number of instances missing a value:
   2.             41
   6.             2
   19.            4
   20.            4
   22.            2
   23.            2
   26.            4

### Source Information
   -- Creator/Donor: Jeffrey C. Schlimmer (Jeffrey.Schlimmer@a.gp.cs.cmu.edu)
   -- Date: 19 May 1987
   -- Sources:
     1) 1985 Model Import Car and Truck Specifications, 1985 Ward's
        Automotive Yearbook.
     2) Personal Auto Manuals, Insurance Services Office, 160 Water
        Street, New York, NY 10038 
     3) Insurance Collision Report, Insurance Institute for Highway
        Safety, Watergate 600, Washington, DC 20037

### Past Usage
   -- Kibler,~D., Aha,~D.~W., \& Albert,~M. (1989).  Instance-based prediction
      of real-valued attributes.  {\it Computational Intelligence}, {\it 5},
      51--57.
	 -- Predicted price of car using all numeric and Boolean attributes
	 -- Method: an instance-based learning (IBL) algorithm derived from a
	    localized k-nearest neighbor algorithm.  Compared with a
	    linear regression prediction...so all instances
	    with missing attribute values were discarded.  This resulted with
	    a training set of 159 instances, which was also used as a test
	    set (minus the actual instance during testing).
	 -- Results: Percent Average Deviation Error of Prediction from Actual
	    -- 11.84% for the IBL algorithm
	    -- 14.12% for the resulting linear regression equation

4. Relevant Information:
   -- Description
      This data set consists of three types of entities: (a) the
      specification of an auto in terms of various characteristics, (b)
      its assigned insurance risk rating, (c) its normalized losses in use
      as compared to other cars.  The second rating corresponds to the
      degree to which the auto is more risky than its price indicates.
      Cars are initially assigned a risk factor symbol associated with its
      price.   Then, if it is more risky (or less), this symbol is
      adjusted by moving it up (or down) the scale.  Actuarians call this
      process "symboling".  A value of +3 indicates that the auto is
      risky, -3 that it is probably pretty safe.

      The third factor is the relative average loss payment per insured
      vehicle year.  This value is normalized for all autos within a
      particular size classification (two-door small, station wagons,
      sports/speciality, etc...), and represents the average loss per car
      per year.

   -- Note: Several of the attributes in the database could be used as a
            "class" attribute.

        '''
        st.write(description)
    else:
        st.header('Automobile dataset')
        df = load_data('../data/automobile_preprocessed.csv')
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
            epochs = st.slider('Select number of epochs', min_value=1, max_value=1000, value=1, step=3, key='epochs')
            parameters['epochs'] = epochs

        X_train, y_train, X_test, y_test = prepared_dataset(df)
        if model_type == 'neural_network':
            input_shape = X_train.shape[1]
            parameters['input_shape'] = input_shape
        model = trainee_model(X_train, y_train, model=model_type, parameters=parameters)
        if model_type != 'neural_network':
            st.write(model)
            # Сериализация модели
            model_bytes = serialize_model(model)
            # Десериализация модели
            model_deser = deserialize_model(model_bytes)
        process_prediction(model, X_test, y_test)


# In[13]:


if __name__ == "__main__":
    main()


# In[ ]:




