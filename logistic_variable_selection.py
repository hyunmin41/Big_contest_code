import statsmodels.api as sm 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf

data=pd.read_csv("~\\bigcon_data_6_v.csv")
data1=pd.read_excel("~\\bigcon_data_1_v.xlsx")
data2=pd.read_csv("~\\bigcon_data_2_v.csv")
data3=pd.read_csv("~\\bigcon_data_3_v.csv")
data4=pd.read_csv("~\\bigcon_data_4_v.csv")
data5=pd.read_csv("~\\bigcon_data_5_v.csv")

#집단1
import random
seed = 1234
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

for i in data.select_dtypes('object').columns:
    le = LabelEncoder().fit(data[i])
    data[i] = le.transform(data[i])
    
X_scaler = MinMaxScaler()
Y_scaler = MinMaxScaler()
X_data = X_scaler.fit_transform(data1[['유역평균강수','강우(A지역)', '강우(B지역)', '강우(C지역)', '강우(D지역)','수위(D지역)','수위(E지역)']])
Y_data = Y_scaler.fit_transform(data1[['유입량']]) 

def custom_ts_multi_data_prep(dataset, target, start, end, window, horizon):
    X = []
    y = []
    start = start + window
    if end is None:
        end = len(dataset) - horizon
    for i in range(start, end):
        indices = range(i-window, i)
        X.append(dataset[indices])
        indicey = range(i+1, i+1+horizon)
        y.append(target[indicey])
    return np.array(X), np.array(y) 

X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.33, random_state=42)

model = sm.Logit(endog=y_train, exog=X_train).fit()
print(model.summary())

#집단2
import random
seed = 1234
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

for i in data2.select_dtypes('object').columns:
    le = LabelEncoder().fit(data2[i])
    data2[i] = le.transform(data2[i])
    
X_scaler = MinMaxScaler()
Y_scaler = MinMaxScaler()
X_data = X_scaler.fit_transform(data2[['유역평균강수','강우(A지역)', '강우(B지역)', '강우(C지역)', '강우(D지역)','수위(D지역)','수위(E지역)']])
Y_data = Y_scaler.fit_transform(data2[['유입량']]) 

def custom_ts_multi_data_prep(dataset, target, start, end, window, horizon):
    X = []
    y = []
    start = start + window
    if end is None:
        end = len(dataset) - horizon
    for i in range(start, end):
        indices = range(i-window, i)
        X.append(dataset[indices])
        indicey = range(i+1, i+1+horizon)
        y.append(target[indicey])
    return np.array(X), np.array(y) 

X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.33, random_state=42)

model = sm.Logit(endog=y_train, exog=X_train).fit()
print(model.summary())

#집단3
import random
seed = 1234
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

for i in data3.select_dtypes('object').columns:
    le = LabelEncoder().fit(data3[i])
    data3[i] = le.transform(data3[i])
    
X_scaler = MinMaxScaler()
Y_scaler = MinMaxScaler()
X_data = X_scaler.fit_transform(data3[['유역평균강수','강우(A지역)', '강우(B지역)', '강우(C지역)', '강우(D지역)','수위(D지역)','수위(E지역)']])
Y_data = Y_scaler.fit_transform(data3[['유입량']]) 

def custom_ts_multi_data_prep(dataset, target, start, end, window, horizon):
    X = []
    y = []
    start = start + window
    if end is None:
        end = len(dataset) - horizon
    for i in range(start, end):
        indices = range(i-window, i)
        X.append(dataset[indices])
        indicey = range(i+1, i+1+horizon)
        y.append(target[indicey])
    return np.array(X), np.array(y) 

X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.33, random_state=42)

model = sm.Logit(endog=y_train, exog=X_train).fit()
print(model.summary())

#집단4
import random
seed = 1234
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

for i in data4.select_dtypes('object').columns:
    le = LabelEncoder().fit(data4[i])
    data4[i] = le.transform(data4[i])
    
X_scaler = MinMaxScaler()
Y_scaler = MinMaxScaler()
X_data = X_scaler.fit_transform(data4[['유역평균강수','강우(A지역)', '강우(B지역)', '강우(C지역)', '강우(D지역)','수위(D지역)','수위(E지역)']])
Y_data = Y_scaler.fit_transform(data4[['유입량']]) 

def custom_ts_multi_data_prep(dataset, target, start, end, window, horizon):
    X = []
    y = []
    start = start + window
    if end is None:
        end = len(dataset) - horizon
    for i in range(start, end):
        indices = range(i-window, i)
        X.append(dataset[indices])
        indicey = range(i+1, i+1+horizon)
        y.append(target[indicey])
    return np.array(X), np.array(y) 

X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.33, random_state=42)

model = sm.Logit(endog=y_train, exog=X_train).fit()
print(model.summary())

#집단5
import random
seed = 1234
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

for i in data.select_dtypes('object').columns:
    le = LabelEncoder().fit(data5[i])
    data5[i] = le.transform(data5[i])
    
X_scaler = MinMaxScaler()
Y_scaler = MinMaxScaler()
X_data = X_scaler.fit_transform(data5[['유역평균강수','강우(A지역)', '강우(B지역)', '강우(C지역)', '강우(D지역)','수위(D지역)','수위(E지역)']])
Y_data = Y_scaler.fit_transform(data5[['유입량']]) 

def custom_ts_multi_data_prep(dataset, target, start, end, window, horizon):
    X = []
    y = []
    start = start + window
    if end is None:
        end = len(dataset) - horizon
    for i in range(start, end):
        indices = range(i-window, i)
        X.append(dataset[indices])
        indicey = range(i+1, i+1+horizon)
        y.append(target[indicey])
    return np.array(X), np.array(y) 

X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.33, random_state=42)

model = sm.Logit(endog=y_train, exog=X_train).fit()
print(model.summary())

#집단6
import random
seed = 1234
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

for i in data.select_dtypes('object').columns:
    le = LabelEncoder().fit(data[i])
    data[i] = le.transform(data[i])
    
X_scaler = MinMaxScaler()
Y_scaler = MinMaxScaler()
X_data = X_scaler.fit_transform(data[['유역평균강수','강우(A지역)', '강우(B지역)', '강우(C지역)', '강우(D지역)','수위(D지역)','수위(E지역)']])
Y_data = Y_scaler.fit_transform(data[['유입량']]) 

def custom_ts_multi_data_prep(dataset, target, start, end, window, horizon):
    X = []
    y = []
    start = start + window
    if end is None:
        end = len(dataset) - horizon
    for i in range(start, end):
        indices = range(i-window, i)
        X.append(dataset[indices])
        indicey = range(i+1, i+1+horizon)
        y.append(target[indicey])
    return np.array(X), np.array(y) 

X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.33, random_state=42)

model = sm.Logit(endog=y_train, exog=X_train).fit()
print(model.summary())
sorted(y_train,reverse=True)
