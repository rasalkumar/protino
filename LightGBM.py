import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgbm
import numpy as np
import pandas as pd

def main():

    params = {"objective" : "binary", 
              "metric" : "binary_logloss", 
              "max_depth": 11}
    #           "min_child_samples": 20, 
    #           "reg_alpha": 0.2, 
    #           "reg_lambda": 0.2,
    #           "num_leaves" : 100, 
    #           "learning_rate" : 0.01, 
    #           "subsample" : 0.9, 
    #           "colsample_bytree" : 0.9, 
    #           "subsample_freq ": 5}

    n_fold = 10

    data         = pd.read_csv("./query_large.csv", sep=",", parse_dates=['time'], squeeze=True)
    data['time'] = data['time'].apply(lambda y: y.timestamp())
    X = data[['time', 'latitude', 'longitude']]
    y = data[['mag']]

    y.mag = y.mag.apply(lambda x: 1 if x>4.3 else 0)
    X_train, y_train, X_test, y_test = X[:-1000], y[:-1000], X[-1000:], y[-1000:]


    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score

    # splitting data into training and validation set
    xtrain, xvalid, ytrain, yvalid = X_train, X_test, y_train, y_test

    folds = KFold(n_splits=n_fold, shuffle=False, random_state=42)
    prediction = np.zeros(X_test.shape[0])

    d_train = lgbm.Dataset(xtrain, ytrain)
    d_valid = lgbm.Dataset(xvalid, yvalid)

    for i in range(25):

        print('Fold:', i)

        model = lgbm.train(params, d_train,5000, valid_sets=[d_valid], verbose_eval=50, early_stopping_rounds=1000)

    prediction = model.predict(xvalid) # predicting on the validation set
    prediction_int = prediction >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0
    prediction_int = prediction_int.astype(np.int)
    print("Validation F1 Score :")
    print(f1_score(yvalid, prediction_int)) # calculating f1 score

if __name__ == '__main__':
    main()
    