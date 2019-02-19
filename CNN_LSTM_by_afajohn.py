# https://www.kaggle.com/afajohn/cnn-lstm-for-signal-classification-lb-0-513

import keras
import keras.backend as K
from keras.layers import LSTM,Dropout,Dense,TimeDistributed,Conv1D,MaxPooling1D,Flatten
from keras.models import Sequential
import tensorflow as tf
import gc
from numba import jit
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sns.set_style("whitegrid")

# 1. Load Data
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

train_set = pq.read_pandas('train.parquet').to_pandas()
meta_train = pd.read_csv('metadata_train.csv')

print("-------------train_set-------------")
train_set.info()
print("-------------meta_train-------------")
meta_train.info()

# 2. Process and Minimize Data
@jit('float32(float32[:,:], int32)')
def feature_extractor(x, n_part=1000):
    lenght = len(x)
    pool = np.int32(np.ceil(lenght/n_part))
    output = np.zeros((n_part,))
    for j, i in enumerate(range(0,lenght, pool)):
        if i+pool < lenght:
            k = x[i:i+pool]
        else:
            k = x[i:]
        output[j] = np.max(k, axis=0) - np.min(k, axis=0)
    return output

x_train = []
y_train = []
for i in tqdm(meta_train.signal_id):
    idx = meta_train.loc[meta_train.signal_id==i, 'signal_id'].values.tolist()
    y_train.append(meta_train.loc[meta_train.signal_id==i, 'target'].values)
    x_train.append(abs(feature_extractor(train_set.iloc[:, idx].values, n_part=400)))

del train_set; gc.collect()

y_train = np.array(y_train).reshape(-1,)
X_train = np.array(x_train).reshape(-1,x_train[0].shape[0])

print("-------------y_train.shape-------------")
print(y_train.shape)
print("-------------X_train.shape-------------")
print(X_train.shape)

# 3. Build Primitive CNN + LSTM Model
def keras_auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

n_signals = 1 #So far each instance is one signal. We will diversify them in next step
n_outputs = 1 #Binary Classification

#Build the model
verbose, epochs, batch_size = True, 1, 16
n_steps, n_length = 40, 10
X_train = X_train.reshape((X_train.shape[0], n_steps, n_length, n_signals))
print("-------------New X_train.shape-------------")
print(X_train.shape)

# define model
model = Sequential()
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None,n_length,n_signals)))
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
model.add(TimeDistributed(Dropout(0.5)))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(100))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[keras_auc])

model.summary()

for layer in model.layers:
    print(layer.input_shape)

model_callback = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

dir(model_callback)


