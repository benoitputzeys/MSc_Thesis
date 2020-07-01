import numpy as np
from scipy.ndimage.interpolation import shift
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

def create_dates(features_df, y_values):

    date_list = [datetime.datetime(year=int(features_df[i, -1]),
                                   month=int(features_df[i, -2]),
                                   day=int(round(features_df[i, -3])),
                                   hour=int((features_df[i, -4] - 1) / 2),
                                   minute=(i % 2) * 30) for i in range(len(features_df))]
    df_dates = DataFrame(date_list, columns=['Date'])
    df_dates = df_dates.set_index(['Date'])
    df_dates['Load'] = y_values

    return df_dates

########################################################################################################################
# Importing the data.
########################################################################################################################

from numpy import genfromtxt

# Get the X (containing the features) and y (containing the labels) values
X = genfromtxt('C:\Python\Pycharm\MSc_Thesis\Data_Entsoe\Data_Preprocessing\For_Multi_Step_Prediction_Outside_Test_Set\X.csv', delimiter=',')
y = genfromtxt('C:\Python\Pycharm\MSc_Thesis\Data_Entsoe\Data_Preprocessing\For_Multi_Step_Prediction_Outside_Test_Set\y.csv', delimiter=',')
y = np.reshape(y, (len(y), 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = False)
#X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(X_train, y_train, test_size = 0.5, random_state = 0, shuffle = False)

########################################################################################################################
# Set the hyperparameters and fit the model.
########################################################################################################################

import pmdarima as pm

stepwise_model = pm.auto_arima(X_train_1[:,0], start_p=1, start_q=1,
                           max_p=3, max_q=3, m=48,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)
print(stepwise_model.aic())

print(stepwise_model.summary())


#predictions_train_2 = model_fit.predict(len(X_train_2), exog=X_train_2[:,1])
predictions_train_1 = stepwise_model.predict(n_periods=48)
predictions_train_1 = np.reshape(predictions_train_1,(-1,1))
residuals = y_train_1 - predictions_train_1

# Get the errors.
print("-"*200)
print("The mean absolute error of the test set is %0.2f" % np.average(abs(residuals)))
print("The mean squared error of the test set is %0.2f" % np.average(abs(residuals)**2))
print("The root mean squared error of the test set is %0.2f" % np.sqrt(np.mean(abs(residuals)**2)))
print("The mean absolute percent error of the test set is %0.2f" % np.mean(abs((y_train_1-predictions_train_1)/y_train_1)))
print("-"*200)

########################################################################################################################
# Plot the prediction on train set 1.
########################################################################################################################

fig, ax = plt.subplots(2)
fig.suptitle('SARIMA Model', fontsize=20)
#y_values_dates = create_dates(X_train_2[-48*7:], y_train_2[-48*7:])
#ax[0].plot(y_values_dates, label='Train Values')
#y_values_dates = create_dates(X_train_2[-48*7:], y_train_2[-48*7:])
ax[0].plot(y_train_2[:48], label='Actual Values')
#y_values_dates = create_dates(X_train_2[-48*7:], predictions_train_2[-48*7:])
ax[0].plot(predictions_train_1, label='Predictions')
ax[0].set_xlabel('Settlement Period Train Set 1')
ax[0].set_ylabel('Electricity Load [MW]')
ax[0].legend(loc="lower right")
plt.show()

#y_values_dates = create_dates(X_train_2[-48*7:],residuals)
ax[1].plot(residuals, color = 'black', label='Residuals')
ax[1].axhline(0, linestyle='--', color='k')
ax[1].set_xlabel('Settlement Period Train Set 1')
ax[1].set_ylabel('Electricity Load [MW]')
ax[1].legend(loc="lower right")
plt.show()

########################################################################################################################
# Plot the prediction on the train set 2.
########################################################################################################################

#predictions_train_2 = model_fit.predict(len(X_train_2), exog=X_train_2[:,1])
predictions_train_2 = stepwise_model.predict(start = len(X_train_1), end = len(X_train_1)+len(X_train_2)-1)
predictions_train_2 = np.reshape(predictions_train_2,(-1,1))
residuals = y_train_2 - predictions_train_2

# Get the errors.
print("-"*200)
print("The mean absolute error of the train set 2 is %0.2f" % np.average(abs(residuals)))
print("The mean squared error of the train set 2 is %0.2f" % np.average(abs(residuals)**2))
print("The root mean squared error of the train set 2 is %0.2f" % np.sqrt(np.mean(abs(residuals)**2)))
print("The mean absolute percent error of the train set 2 is %0.2f" % np.mean(abs((y_train_2[-48*7:]-predictions_train_2[-48*7:])/y_train_2[-48*7:])))
print("-"*200)

########################################################################################################################
# Plot the prediction of train set 2.
########################################################################################################################

fig, ax = plt.subplots(2)
fig.suptitle('SARIMA Model', fontsize=20)
#y_values_dates = create_dates(X_train_2[-48*7:], y_train_2[-48*7:])
#ax[0].plot(y_values_dates, label='Train Values')
#y_values_dates = create_dates(X_train_2[-48*7:], y_train_2[-48*7:])
ax[0].plot(y_train_2, label='Actual Values')
#y_values_dates = create_dates(X_train_2[-48*7:], predictions_train_2[-48*7:])
ax[0].plot(predictions_train_2, label='Predictions')
ax[0].set_xlabel('Settlement Period Train Set 2')
ax[0].set_ylabel('Electricity Load [MW]')
ax[0].legend(loc="lower right")
plt.show()

#y_values_dates = create_dates(X_train_2[-48*7:],residuals)
ax[1].plot(residuals, color = 'black', label='Residuals')
ax[1].axhline(0, linestyle='--', color='k')
ax[1].set_xlabel('Settlement Period Train Set 2')
ax[1].set_ylabel('Electricity Load [MW]')
ax[1].legend(loc="lower right")
plt.show()
