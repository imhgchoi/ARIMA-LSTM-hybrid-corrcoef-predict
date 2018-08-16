import pandas as pd
import numpy as np
import os
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l1_l2

dataset = pd.read_csv('C:/Users/Froilan/Desktop/myFiles/JupyterFiles/stock_correlation_prediction/train_dev_test/new_asset_after_arima.csv')
dataset = dataset.loc[:,~dataset.columns.str.contains('^Unnamed')]
X = dataset.loc[:,~dataset.columns.str.contains('20')]
Y = dataset.loc[:,dataset.columns.str.contains('20')]

X = np.asarray(X).reshape(180,20,1)
Y = np.asarray(Y).reshape(180,1)


#define custom activation
class Double_Tanh(Activation):
    def __init__(self, activation, **kwargs):
        super(Double_Tanh, self).__init__(activation, **kwargs)
        self.__name__ = 'double_tanh'

def double_tanh(x):
    return (K.tanh(x) * 2)

get_custom_objects().update({'double_tanh':Double_Tanh(double_tanh)})



model = load_model('C:/Users/Froilan/Desktop/myFiles/JupyterFiles/stock_correlation_prediction/models/hybrid_LSTM/epoch28.h5')
score = model.evaluate(X,Y)
print('score : mse - ' + str(score[1]) + ' / mae - ' + str(score[2]))