import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
from tqdm import tqdm
from pylab import rcParams
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters
from torch import nn, optim

#################
### load data ###
#################

df = pd.read_csv("~\\bigcon_data_all.csv", header=0,engine='python')  

df = df.loc[1:3051]
df = df[["유입량",
         "데이터집단 1","Unnamed: 7","Unnamed: 8","Unnamed: 9","Unnamed: 10","Unnamed: 11","Unnamed: 12",
         "데이터집단 2","Unnamed: 14","Unnamed: 15", "Unnamed: 16","Unnamed: 17","Unnamed: 18","Unnamed: 19",
         "데이터집단 3","Unnamed: 21","Unnamed: 22","Unnamed: 23","Unnamed: 24","Unnamed: 25","Unnamed: 26",
         "데이터 집단 4","Unnamed: 28","Unnamed: 29","Unnamed: 30","Unnamed: 31","Unnamed: 32","Unnamed: 33",
         "데이터 집단 5","Unnamed: 35","Unnamed: 36","Unnamed: 37","Unnamed: 38","Unnamed: 39","Unnamed: 40",
         "데이터 집단 6","Unnamed: 42","Unnamed: 43","Unnamed: 44","Unnamed: 45","Unnamed: 46","Unnamed: 47" ]]
df.columns = ["amount",
              "average1", "A1", "B1", "C1", "D1", "EH1", "DH1",
              "average2", "A2", "B2", "C2", "D2", "EH2", "DH2",
              "average3", "A3", "B3", "C3", "D3", "EH3", "DH3",
              "average4", "A4", "B4", "C4", "D4", "EH4", "DH4",
              "average5", "A5", "B5", "C5", "D5", "EH5", "DH5",
              "average6", "A6", "B6", "C6", "D6", "EH6", "DH6"]
df.head()

feature = df[["average5", "A5", "B5", "C5", "D5", "EH5", "DH5"]]
label = df[["amount"]]

################
### modeling ###
################

from sklearn.model_selection import train_test_split

feature_cols = ["average5", "A5", "B5", "C5", "D5", "EH5", "DH5"]
label_cols = ['amount']

from sklearn.preprocessing import MinMaxScaler

feature.sort_index(ascending=False).reset_index(drop=True)

scaler = MinMaxScaler()

feature = scaler.fit_transform(feature)
feature = pd.DataFrame(feature)
feature.columns = feature_cols

train_size = 2023
test_size = 868
train_feature, test_feature = feature[0:train_size], feature[train_size:2891]
train_label, test_label = label[0:train_size], label[train_size:2891]
pred_feature, pred_label = feature[2881:], label[2881:]
print(len(train_feature), len(test_feature), len(train_label), len(test_label), len(pred_feature), len(pred_label))

def make_dataset(df, label, window_size=10):
    feature_list = []
    label_list = []
    for i in range(len(df) - window_size):
        feature_list.append(np.array(df.iloc[i:i+window_size]))
        label_list.append(np.array(label.iloc[i+window_size]))
    return np.array(feature_list), np.array(label_list)

train_feature, train_label = make_dataset(train_feature, train_label, 10)

x_train, x_valid, y_train, y_valid = train_test_split(train_feature, train_label, test_size=0.2)
x_train.shape, x_valid.shape, y_train.shape, y_valid.shape

test_feature.shape, test_label.shape
pred_feature.shape, pred_label.shape

test_feature, test_label = make_dataset(test_feature, test_label, 10)
test_feature.shape, test_label.shape
pred_feature, pred_label = make_dataset(pred_feature, pred_label)
pred_feature.shape, pred_label.shape

subsequences = 2
timesteps = x_train.shape[1]//subsequences
X_train_series_sub = x_train.reshape((x_train.shape[0], subsequences, timesteps, 7)) #7은 들어가는 변수 개수
X_valid_series_sub = x_valid.reshape((x_valid.shape[0], subsequences, timesteps, 7)) #7은 들어가는 변수 개수
print('Train set shape', X_train_series_sub.shape)
print('Validation set shape', X_valid_series_sub.shape)

test_feature_series_sub = test_feature.reshape((test_feature.shape[0], subsequences, timesteps, 7 )) #7은 들어가는 변수 개수

pred_feature_series_sub = pred_feature.reshape((pred_feature.shape[0], subsequences, timesteps, 7 )) #7은 들어가는 변수 개수

# for reproducibility
import tensorflow as tf
import random
seed = 1234
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D, Flatten, TimeDistributed

model_cnn_lstm = Sequential()
model_cnn_lstm.add(TimeDistributed(Conv1D(filters=32, kernel_size=2, 
                                          activation='relu'),
                                   input_shape=(None, X_train_series_sub.shape[2],
                                                X_train_series_sub.shape[3])))
model_cnn_lstm.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model_cnn_lstm.add(TimeDistributed(Dropout((0.5))))
model_cnn_lstm.add(TimeDistributed(Flatten()))
model_cnn_lstm.add(LSTM(120, activation='tanh'))
model_cnn_lstm.add(Dropout(0.2))
model_cnn_lstm.add(Dense(60))
model_cnn_lstm.add(Dense(20))
model_cnn_lstm.add(Dense(1))
model_cnn_lstm.compile(loss='mse', optimizer="adam")

import os
import keras

early_stop = EarlyStopping(monitor='val_loss', patience=5)

model_path = 'model'
filename = os.path.join(model_path, 'tmp_checkpoint.h5')
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

history = model_cnn_lstm.fit(X_train_series_sub, y_train,
                    epochs=40, steps_per_epoch=50,
                    verbose=1, 
                    validation_data=(X_valid_series_sub, y_valid), 
                    callbacks=[early_stop, checkpoint])

######################
### Model accuracy ###
######################

model_cnn_lstm.load_weights(filename)
test_pred = model_cnn_lstm.predict(test_feature_series_sub)

# model RMSE
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE : ', RMSE(test_label, test_pred)) 

plt.figure(figsize=(12, 9))
plt.plot(test_label, label = 'actual')
plt.plot(test_pred, label = 'prediction')
plt.legend()
plt.show()

#######################
### CNN forecasting ###
#######################

# 홍수 사상 26번 160개 예측
pred = model_cnn_lstm.predict(pred_feature_series_sub)
pred

# save the predicted data
prediction=pd.DataFrame(pred)
file_name='데이터집단5_CNN_excepted.xlsx'
prediction.to_excel(file_name)

plt.figure(figsize=(12, 9))
plt.plot(pred, label = 'prediction')
plt.legend()
plt.show()

###########################
### plot whole y values ###
###########################

a=pd.DataFrame(df[0:2891]["amount"])
b=pd.DataFrame(pred, columns = ["amount"])
mod_df = a.append(b.loc[:], ignore_index=True)
mod_df

plt.figure(figsize=(16,9))
plt.plot(mod_df)
plt.title("time series")
plt.ylabel("total influx")
plt.show()