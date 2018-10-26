import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk
import warnings 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
warnings.filterwarnings("ignore", category=DeprecationWarning)
def main():
    data         = pd.read_csv("./query_large.csv", sep=",", parse_dates=['time'], squeeze=True)
    data['time'] = data['time'].apply(lambda y: y.timestamp())
    X = data[['time', 'latitude', 'longitude']]
    y = data[['mag']]

    y.mag = y.mag.apply(lambda x: 1 if x>4.3 else 0)
    xtrain, ytrain, xvalid, yvalid = X[:-1000], y[:-1000], X[-1000:], y[-1000:]

    lreg = LogisticRegression(verbose=1)
    lreg.fit(xtrain, ytrain) # training the model

    prediction = lreg.predict_proba(xvalid) # predicting on the validation set
    prediction_int = prediction[:,1] >= 0.2 # if prediction is greater than or equal to 0.3 than 1 else 0
    prediction_int = prediction_int.astype(np.int)

    print(f1_score(yvalid.mag.values, prediction_int)) # calculating f1 score
    
    test_pred = lreg.predict_proba(xvalid)
    test_pred_int = test_pred[:,1] >= 0.2
    test_pred_int = test_pred_int.astype(np.int)
    submission = xvalid.copy()
    submission['mag'] = test_pred_int
    submission.to_csv('submission_logistic_0.csv', index=False) # writing data to a CSV file
if __name__ == '__main__':
    main()    