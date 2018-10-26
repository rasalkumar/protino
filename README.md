# Temblor Predict

Temblor predict is a tool for predicting the next earthquake magnitude, by capturing the pattern of earthquake magnitude in a particular region. 

Earthquakes is one of the most devastating natural calamity that takes thousand of lives and leaves millions more homeless and deprives them of basic utility.

## Dataset

  - Data is obtained from United States Geological Survey [1], the latitude range for the earthquake occurences is 20°S and 40°S and longitude range is 70°E to 105°E, from the year 1973
  <center><img src="./dataset.png"></center>

## Techniques and Results 

### Nowcasting models

As small tremors are indicator for stress building up in a region. In this technique we capture the time of occurrence of large earthquakes using the pattern of small tremors that occur between two large earthquakes. 

 - The system then uses
   - Time Delay model
   - Seasonal Autoregressive Integrated Moving Average model
   - Long Short Term Memory model
   
    
to predict number of small earthquakes before a large magnitude (>5) earthquake occurs. 

1. Time Delay Neural Network Implementation(TDNN): The TDNN is trained on a time-series indicating the number of tremors between two large magnitude earthquake(>5 magnitude). The model is trained on data from 1973 to 2018. To predict the next number in the sequence(indicating the number of tremors before large earthquake), the user has to give the last 5 number of sequence.
2. SARIMAX model was also trained similarly, it was trained to predict the number of tremors before the next big earthquake. In wrapper file, that is, runner.ipynb we calculate the number of small earthquakes between the interval given.  

 3. LSTM is also a time series prediction neural network, which achieves the same task of predicting the number of tremors before the next big earthquake.

### Regression models

We use Binary classification in all our regression models to calculate the probability of occurrence of earthquake. If the magnitude of the earthquake is more than 4.3 on the Richter scale we consider the the output to be 1 otherwise we consider the output to be 0. We split the data into training and validation sets such that the validation set consists of last 1000 elements of the data set. The hyperparameters were tuned for various models using GridSearch To calculate the best possible accuracy. The results on the data are saved in the file submission.csv and the accuracy of the regression models is recorded.

The regression models such such as Logistic Regression, Random Forest and LightGBM give an accuracy of about 65 to 67 percent.


## Tech

Temblor Predict uses a number of open source projects to work properly:

* [Tensorflow](https://github.com/tensorflow/tensorflow) - Deep Learning Library for python
* [Keras](https://github.com/keras-team/keras) - wrapper for tensorflow
* [Pandas](https://github.com/pandas-dev/pandas) - Python package for expressive data structures
* [Matplotlib](https://github.com/matplotlib/matplotlib) - Matplotlib is a Python 2D plotting library 

### Training the dataset
The training code is given in the form of IPython notebooks
1. Nowcasting_SARIMAX.ipynb
2. Nowcasting_LSTM.ipynb
3. Tdnn.ipynb
4. SARIMAX.ipynb
5. Random Forest .ipynb
6. Logistic Regression.ipynb
7. LightGBM.ipynb

## Usage
runner.ipynb is a wrapper notebook which is calls the prediction function for the trained models, which were trained using the above given notebooks. The user needs to supply the necessary arguments as given in comments in runner.ipynb

