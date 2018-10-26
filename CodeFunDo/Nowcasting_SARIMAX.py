import sys
import pandas as pd
import numpy as np
import datetime
from datetime import datetime, date
import math
import tensorflow as tf
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMAResults
import matplotlib.pyplot as plt
RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)

def main():

    data = pd.read_csv("./query_large.csv", sep=",")
    np_date = ["" for i in range(len(data))]

    data = data[296:]
    data = data.iloc[::-1]
    for i,item in enumerate(data["time"]):
        np_date[i] = datetime.strptime(item[:-5], '%Y-%m-%dT%H:%M:%S')
    X = np.array(data["mag"])
    for i in range(len(X)):
        if X[i] >= 5:
            X = X[i:]
            break
    count = 1
    ans = []
    new_data = []
    for i in range(len(X)):
        if X[i] < 5:
            #print('in if ' + str( X[i]))
            count = count + 1
        else:
           # print('in else ' + str( X[i]))
           # print('appended '+ str(count))
            ans.append(count)
            count = 0
            #new_data.append(data["time"][i])
    data_f = pd.DataFrame(data = ans, columns = ['timeSinceLast'])

    dataset = data_f
    model = sm.tsa.statespace.SARIMAX(dataset, order=(5,0,1))
    model_fit = model.fit(disp=0)

    model_fit.save('arima_nowcasting.pkl')

    model = ARIMAResults.load('arima_nowcasting.pkl')

    start_index = int(sys.argv[1])
    end_index   = int(sys.argv[2])
    forecast = model.predict(start=start_index, end=end_index)
    print(forecast)
    plt.scatter(forecast.index.tolist(), forecast.values);

if __name__ == '__main__':
    main()
    