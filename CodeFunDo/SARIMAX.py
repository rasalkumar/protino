import time
import datetime
from statsmodels.tsa.arima_model import ARIMAResults
import pandas as pd
import numpy as np
import datetime
from datetime import datetime, date
import math
from sklearn.utils import resample
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import sys
# import tensorflow as tf
# RANDOM_SEED = 42
# tf.set_random_seed(RANDOM_SEED)

def main():

    data = pd.read_csv("./query_large.csv", sep=",", parse_dates=['time'], index_col='time', squeeze=True) #, date_parser=parser)

    fit = data['mag']
    model = sm.tsa.statespace.SARIMAX(fit, order=(1,0,1))
    model_fit = model.fit(disp=0)

    residuals = DataFrame(model_fit.resid)

    model_fit.save('arima_normal.pkl')
    timestamp = lambda s : datetime.strptime(s, "%d/%m/%Y")

    model = ARIMAResults.load('arima_normal.pkl')

    start_index = int(sys.argv[1])
    
    end_index = start_index + 6    #datetime(1925, 12, 26)
    forecast = model.predict(start=start_index, end=end_index)
    print(forecast)
if __name__ == '__main__':
    main()



