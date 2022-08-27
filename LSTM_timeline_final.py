import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf
import pandas_profiling

#################
### load data ###
#################

data1 = pd.read_csv("~\\bigcon_data_1.csv", header=0,engine='python')
data2 = pd.read_csv("~\\bigcon_data_2.csv", header=0,engine='python')
data3 = pd.read_csv("~\\bigcon_data_3.csv", header=0,engine='python')
data4 = pd.read_csv("~\\bigcon_data_4.csv", header=0,engine='python')
data5 = pd.read_csv("~\\bigcon_data_5.csv", header=0,engine='python')
data6 = pd.read_csv("~\\bigcon_data_6.csv", header=0,engine='python')

#######################################
### prove similarity of correlation ###
#######################################

average=pd.concat([data1["유역평균강수"],data2["유역평균강수"],data3["유역평균강수"],data4["유역평균강수"],data5["유역평균강수"],data6["유역평균강수"]], axis=1)
rainA=pd.concat([data1["강우(A지역)"],data2["강우(A지역)"],data3["강우(A지역)"],data4["강우(A지역)"],data5["강우(A지역)"],data6["강우(A지역)"]], axis=1)
rainB=pd.concat([data1["강우(B지역)"],data2["강우(B지역)"],data3["강우(B지역)"],data4["강우(B지역)"],data5["강우(B지역)"],data6["강우(B지역)"]], axis=1)
rainC=pd.concat([data1["강우(C지역)"],data2["강우(C지역)"],data3["강우(C지역)"],data4["강우(C지역)"],data5["강우(C지역)"],data6["강우(C지역)"]], axis=1)
rainD=pd.concat([data1["강우(D지역)"],data2["강우(D지역)"],data3["강우(D지역)"],data4["강우(D지역)"],data5["강우(D지역)"],data6["강우(D지역)"]], axis=1)
heightE=pd.concat([data1["수위(E지역)"],data2["수위(E지역)"],data3["수위(E지역)"],data4["수위(E지역)"],data5["수위(E지역)"],data6["수위(E지역)"]], axis=1)
heightD=pd.concat([data1["수위(D지역)"],data2["수위(D지역)"],data3["수위(D지역)"],data4["수위(D지역)"],data5["수위(D지역)"],data6["수위(D지역)"]], axis=1)

average.corr();rainA.corr();rainB.corr();rainC.corr();rainD.corr();heightE.corr();heightD.corr()

####################
### profile data ###
####################

data = pd.read_csv("~\\bigcon_data_1.csv", header=0,engine='python')

pr=data.profile_report() 
data.profile_report()
pr.to_file('./data1.html')

################
### modeling ###
################

for i in data.select_dtypes('object').columns:
   le = LabelEncoder().fit(data[i])
   data[i] = le.transform(data[i]) 

X_scaler = MinMaxScaler()
Y_scaler = MinMaxScaler()
X_data = X_scaler.fit_transform(data[['유역평균강수', '강우(A지역)', '강우(C지역)', '수위(E지역)', '수위(D지역)']][0:2891]) 
X_data1 = X_scaler.fit_transform(data[['유역평균강수', '강우(A지역)', '강우(C지역)', '수위(E지역)', '수위(D지역)']])
Y_data = Y_scaler.fit_transform(data[['유입량']][0:2891]) # y value
Y_data1 = Y_scaler.fit_transform(data[['유입량']]) # y value

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

hist_window = 10
horizon = 160 # number of forecasting
TRAIN_SPLIT = 2033 # train set number.

x_train, y_train = custom_ts_multi_data_prep(X_data, Y_data, 0, TRAIN_SPLIT, hist_window, horizon)
x_vali, y_vali = custom_ts_multi_data_prep(X_data, Y_data, TRAIN_SPLIT, None, hist_window, horizon) 

print(len(y_train),len(y_vali))

print ('Multiple window of past history\n')
print(x_train[0])
print ('\n Target horizon\n')
print (y_train[0]) 

# for reproducibility
import random
seed = 1234
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

batch_size = 100
buffer_size = 150
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.cache().shuffle(buffer_size).batch(batch_size).repeat()
val_data = tf.data.Dataset.from_tensor_slices((x_vali, y_vali))
val_data = val_data.batch(batch_size).repeat() 

lstm_model = tf.keras.models.Sequential([
   tf.keras.layers.LSTM(250,input_shape=x_train.shape[-2:]),
     tf.keras.layers.Dense(100, activation='tanh'),
     tf.keras.layers.Dense(50, activation='tanh'),
     tf.keras.layers.Dense(50, activation='tanh'),
     tf.keras.layers.Dropout(0.25),
     tf.keras.layers.Dense(units=horizon),
 ])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.summary() 

model_path = 'LSTM_Multivariate.h5'
early_stopings = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min')
checkpoint =  tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=0)
callbacks=[early_stopings,checkpoint]

history = lstm_model.fit(train_data,epochs=30,steps_per_epoch=50,validation_data=val_data,validation_steps=50,verbose=1,callbacks=callbacks)

plt.figure(figsize=(16,9))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'validation loss'])
plt.show() 

######################
### Model accuracy ###
######################

data_val1 = X_scaler.fit_transform(data[['유역평균강수', '강우(A지역)', '강우(C지역)', '수위(E지역)', '수위(D지역)']][0:2891]) 
val_rescaled1 = data_val1.reshape(1, data_val1.shape[0], data_val1.shape[1])
pred_model = lstm_model.predict(val_rescaled1)
pred_model_Inverse = Y_scaler.inverse_transform(pred_model)
pred_model_Inverse.shape

y_true = Y_scaler.inverse_transform(y_vali[0])

# model RMSE
def timeseries_evaluation_metrics_func(y_true, y_pred):
     def mean_absolute_percentage_error(y_true, y_pred): 
         y_true, y_pred = np.array(y_true), np.array(y_pred)
         return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
     print('Evaluation metric results:-')
     print(f'MSE is : {metrics.mean_squared_error(y_true, y_pred)}')
     print(f'RMSE is : {np.sqrt(metrics.mean_squared_error(y_true, y_pred))}')

timeseries_evaluation_metrics_func(y_true, pred_model_Inverse[0]) 

plt.figure(figsize=(16,9))
plt.plot(y_true)
plt.plot(list(pred_model_Inverse[0]))
plt.title("true vs predicted")
plt.ylabel("Traffic volume")
plt.legend(('true','model predicted'))
plt.show() 

########################
### LSTM forecasting ###
########################

# 홍수 사상 26번 160개 예측
data_val = X_scaler.fit_transform(data[['유역평균강수', '강우(A지역)', '강우(C지역)', '수위(E지역)', '수위(D지역)']]) 
val_rescaled = data_val.reshape(1, data_val.shape[0], data_val.shape[1])
pred = lstm_model.predict(val_rescaled)
pred_Inverse = Y_scaler.inverse_transform(pred)
pred_Inverse 

# save the predicted data
prediction=pd.DataFrame(pred_Inverse)
file_name='파일이름.xlsx'
prediction.to_excel(file_name)

###########################
### plot whole y values ###
###########################
a=pd.DataFrame(data[0:2891]['유입량'])
b=pd.DataFrame(pred_Inverse[0],columns=['유입량'])
mod_df = a.append(b.loc[:], ignore_index=True)

plt.figure(figsize=(16,9))
plt.plot(mod_df)
plt.title("time series")
plt.ylabel("total influx")
plt.show()