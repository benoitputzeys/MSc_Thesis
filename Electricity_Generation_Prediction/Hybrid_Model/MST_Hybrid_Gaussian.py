import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from Electricity_Generation_Prediction.Hybrid_Model.Functions import create_model, train_model, plot_the_loss_curve
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error
import keras
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, RationalQuadratic as RQ, WhiteKernel, ExpSineSquared as Exp, DotProduct as Lin

########################################################################################################################
# Get data and data preprocessing.
########################################################################################################################

from numpy import genfromtxt

# Get the X (containing the features) and y (containing the labels) values
ANN_train = genfromtxt('Electricity_Generation_Prediction/Hybrid_Model/Pred_train2_other_metrics/ANN_prediction.csv', delimiter=',')
RF_train = genfromtxt('Electricity_Generation_Prediction/Hybrid_Model/Pred_train2_other_metrics/RF_prediction.csv', delimiter=',')
DT_train = genfromtxt('Electricity_Generation_Prediction/Hybrid_Model/Pred_train2_other_metrics/DT_prediction.csv', delimiter=',')
SVR_train = genfromtxt('Electricity_Generation_Prediction/Hybrid_Model/Pred_train2_other_metrics/SVR_prediction.csv', delimiter=',')
LSTM_train = genfromtxt('Electricity_Generation_Prediction/Hybrid_Model/Pred_train2_other_metrics/LSTM_prediction.csv', delimiter=',')
SARIMA_train = genfromtxt('Electricity_Generation_Prediction/Hybrid_Model/Pred_train2_other_metrics/SARIMA_prediction.csv', delimiter=',')

ANN_test = genfromtxt('Electricity_Generation_Prediction/Hybrid_Model/Pred_test_other_metrics/ANN_prediction.csv', delimiter=',')
RF_test = genfromtxt('Electricity_Generation_Prediction/Hybrid_Model/Pred_test_other_metrics/RF_prediction.csv', delimiter=',')
DT_test = genfromtxt('Electricity_Generation_Prediction/Hybrid_Model/Pred_test_other_metrics/DT_prediction.csv', delimiter=',')
SVR_test = genfromtxt('Electricity_Generation_Prediction/Hybrid_Model/Pred_test_other_metrics/SVR_prediction.csv', delimiter=',')
LSTM_test = genfromtxt('Electricity_Generation_Prediction/Hybrid_Model/Pred_test_other_metrics/LSTM_prediction.csv', delimiter=',')
SARIMA_test = genfromtxt('Electricity_Generation_Prediction/Hybrid_Model/Pred_test_other_metrics/SARIMA_prediction.csv', delimiter=',')

# Get the X (containing the features) and y (containing the labels) values
X = genfromtxt('Data_Entsoe/Data_Preprocessing/For_Multi_Step_Prediction/X.csv', delimiter=',')
y = genfromtxt('Data_Entsoe/Data_Preprocessing/For_Multi_Step_Prediction/y.csv', delimiter=',')
y = np.reshape(y, (len(y), 1))

# Save the unscaled data for later for data representation.
X = X[0:1000,:]
y = y[0:1000]
X_train, X_test, y_tr, y_te = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = False)

ANN_test = ANN_train[801:1001,0].reshape(-1,1)
ANN_train = ANN_train[1:801,0].reshape(-1,1)

DT_test = DT_train[801:1001,0].reshape(-1,1)
DT_train = DT_train[1:801,0].reshape(-1,1)

RF_test = RF_train[801:1001,0].reshape(-1,1)
RF_train = RF_train[1:801,0].reshape(-1,1)

LSTM_test = LSTM_train[801:1001,0].reshape(-1,1)
LSTM_train = LSTM_train[1:801,0].reshape(-1,1)

SVR_test = SVR_train[801:1001,0].reshape(-1,1)
SVR_train = SVR_train[1:801,0].reshape(-1,1)

SARIMA_test = SARIMA_train[801:1001,0].reshape(-1,1)
SARIMA_train = SARIMA_train[1:801,0].reshape(-1,1)

all_predictions_train = np.concatenate((ANN_train, SVR_train, LSTM_train, DT_train, RF_train), axis = 1)
all_predictions_test = np.concatenate((ANN_test, SVR_test, LSTM_test, DT_test, RF_test), axis = 1)

y_train = np.asarray([y_tr[:,0], ANN_train[:,0], LSTM_train[:,0], RF_train[:,0], SVR_train[:,0], DT_train[:,0]]).T
y_test = np.asarray([y_te[:,0], ANN_test[:,0], LSTM_test[:,0], RF_test[:,0], SVR_test[:,0], DT_test[:,0]]).T

# Feature Scaling
y_scaler = StandardScaler()
all_predictions_train = y_scaler.fit_transform(y_train)
x_scaler = StandardScaler()
x_train = x_scaler.fit_transform(np.atleast_2d(np.linspace(1, len(y_train), len(y_train))).T)
x_test = x_scaler.transform(np.atleast_2d(np.linspace(len(y_train), len(y_train)+ len(y_test), len(y_test))).T)

kernel = RBF(length_scale=48)*Exp(length_scale=48, periodicity=48*7)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)  #Choose your amount of optimizer restarts.
gp.fit(x_train, all_predictions_train)

y_pred_1, sigma_1 = gp.predict(x_test, return_std=True)

y_pred_1 = y_scaler.inverse_transform(y_pred_1)
x_train = x_scaler.inverse_transform(x_train)
x_test = x_scaler.inverse_transform(x_test)

plt.plot(x_train, y[0:800,0], label=u'Training')
plt.plot(x_test, y[800:1001,0], label=u'Observation')
plt.plot(x_test, y_pred_1[:,0], linewidth=1, label=u'Prediction')
lower = y_pred_1 - sigma_1.reshape(-1,1)
upper = y_pred_1 + sigma_1.reshape(-1,1)
plt.fill_between(x_test[:,0],lower[:,0], upper[:,0], alpha=0.2, color='k', label=u'95 % confidence interval')
plt.xlabel('Prediction on the test set.')
plt.legend(loc='upper right', fontsize=10)
