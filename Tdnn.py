import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import pandas as pd
from datetime import datetime,date
from sklearn.model_selection import train_test_split
import numpy as np
import keras
from keras.layers import * 
from keras.models import Sequential 
np.set_printoptions(threshold=np.nan)
import sys
graph = tf.Graph()

def main():


    data = pd.read_csv("/home/LOP/query_large.csv", sep=",")
    np_date = ["" for i in range(len(data))]
    data = data[296:]
    data = data.iloc[::-1]
    # use data after 1973, because before 1973 only above 5 mag earthquake were recorded
    for i, item in enumerate(data["time"]):
        np_date[i] = datetime.strptime(item[:-5], '%Y-%m-%dT%H:%M:%S')
    X = np.array(data["mag"])
    for i in range(len(X)):
        if X[i] >= 5:
            X = X[i:]
            break
    count = 1
    ans = []
    for i in range(len(X)):
        if X[i] < 5:
            count = count + 1
        else:
            ans.append(count)
            count = 0
    data_f = pd.DataFrame(data=ans, columns=['timeSinceLast'])




    data_f = np.array(data_f)

    n_channels = 1
    seq_len =5

    X = []
    y = []
    for i in range(len(data_f)-seq_len-1):
        X.append(np.reshape(data_f[i:i+seq_len], -1, seq_len))
        y.append(data_f[i+seq_len])
    X = np.array(X)
    X = np.reshape(X, (1919, 5, 1))
    y = np.vstack(y)

    xtrain, xval, ytrain, yval = train_test_split(X, y)

    keep_prob_ = 0.2 
    learning_rate_ = 0.001 

    inputs_ = tf.placeholder(tf.float32, [None, seq_len, n_channels],name = 'inputs')
    yplace = tf.placeholder(tf.float32, [None, 1],name = 'ytrue')

    model = Sequential()
    model.add(Conv1D(input_shape = (5,1), filters= 18, kernel_size= 2, strides=1 , padding= 'same', activation= 'relu', name= 'conv1'))
    model.add(MaxPool1D(pool_size= 2, strides= 2, name = 'mxp1')) 
    model.add(Conv1D(filters= 36, kernel_size= 2, strides= 1, padding= 'same', activation= 'relu', name = 'conv2')) 
    model.add(MaxPool1D(pool_size= 2, strides= 2, name= 'mxp2')) 
    # model.add(Conv1D(filters= 72, kernel_size= 2, strides= 1, padding= 'same', activation= 'relu', name='conv3')) 
    # model.add(MaxPool1D(pool_size= 2, strides= 2, name='mxp3')) 
    # model.add(Conv1D(filters= 144, kernel_size= 2, strides= 1, padding= 'same', activation= 'relu', name= 'conv4'))  
    # model.add(MaxPool1D(pool_size= 2, strides= 2, name= 'mxp4')) 
    model.add(Flatten()) 
    model.add(Dropout(0.2))
    model.add(Dense(72)) 
    model.add(Dense(1))  

    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['mse'])
    check = np.array([[int(sys.argv[i+1]) for i in range(5)]]).T  ## checking on custom Data
    print(check)
    check = np.expand_dims(check, axis =0)
    model.input_shape
    print(model.predict(check, batch_size=1))

if __name__ == '__main__':
    main()