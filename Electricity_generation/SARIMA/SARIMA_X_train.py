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

    date_list = [datetime.datetime(year=int(round(features_df[i, -1])),
                                   month=int(round(features_df[i, -2])),
                                   day=int(round(features_df[i, -3])),
                                   hour=int((features_df[i, -4] - 1) / 2),
                                   minute=int(((features_df[i, -4] -1) % 2 ) * 30)) for i in range(len(features_df))]

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
X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(X_train, y_train, test_size = 0.2, random_state = 0, shuffle = False)

ex_variable_train_1 = X_train_1[:,5:]
ex_variable_train_2 = X_train_2[:,5:]
# # Plot the first 7 days in the training set.
# y_values_dates = create_dates(X_train[:7*48], y_train[:7*48])
# plt.figure()
# plt.plot(y_values_dates, linewidth=0.5)
# plt.title('Electricity Load SARIMA Model', fontsize=20)
# plt.ylabel('Electricity Load [MW]', fontsize=16)

########################################################################################################################
# Check for stationarity
########################################################################################################################

# y_1 = y_train[1:]
# y_2 = y_train[:-1]
#
# # Take the difference between the actual and the previous values.
# y_diff = y_1 - y_2
#
# # Plot the difference.
# y_values_dates = create_dates(X_train[:48*7], y_diff[:48*7])
# plt.figure()
# plt.plot(X_train[:,0], linewidth=0.5)
# plt.plot(X_train[:,2], linewidth=0.5)
# plt.title('Electricity Load SARIMA Model', fontsize=20)
# plt.ylabel('Electricity Load [MW]', fontsize=16)
#
# plt.figure()
# plt.plot(X_train[:,0], linewidth=0.5)
# plt.title('Electricity Load SARIMA Model', fontsize=20)
# plt.ylabel('Electricity Load [MW]', fontsize=16)

########################################################################################################################
# Check the auto-correlation and the partial auto-correlation function.
########################################################################################################################

# # Adfuller test
# result = adfuller(y_diff)
# print("ADF statistic: %f" % result[0])
#
# # Print the p-value.
# # If this value is very close to 0, it means it is a positive test in the sense that the series is stationary.
# print("P-value: %f" % result[1])
# print("Critical values:")
#
# for key, value in result[4].items():
#     print("\t%s: %.3f" % (key, value))
#
# # Plot the acf and pacf graphs.
# # Anything within the blue lines is not statistically significant. The blue regions are the error bands.
# # For ACF, we expect a diminishing trend over time.
# fig, ax = plt.subplots(2, figsize=(12,6))
# ax[0] = plot_acf(y_diff,ax=ax[0], lags = 50)
# ax[1] = plot_pacf(y_diff,ax=ax[1], lags = 50)

########################################################################################################################
# Set the hyperparameters and fit the model.
########################################################################################################################

my_order = (0,1,1)
my_seasonal_order = (0, 1, 1, 48)
model = SARIMAX(y_train_1, order=my_order, seasonal_order=my_seasonal_order, exog=ex_variable_train_1)
#model = SARIMAX(y_train_1, order=my_order, seasonal_order=my_seasonal_order)
model_fit = model.fit()

# Summary of the model
print(model_fit.summary())

########################################################################################################################
# Decompose the data into seasonal component, trend and residual error of the 2.
########################################################################################################################

# # Decompose the data.
# ts_decompose = sm.tsa.seasonal_decompose(y_train, model='additive', period = 48)
# ts_decompose.plot()
# plt.show

# Get the prediction and its residual.
# Define the lenght of the prediciton.

#predictions_train_2 = model_fit.predict(len(X_train_2), exog=X_train_2[:,1])
predictions_train_1 = model_fit.predict(start = 1, end = len(X_train_1), exog=ex_variable_train_1)
predictions_train_2 = model_fit.forecast(steps = len(X_train_2), exog=ex_variable_train_2 )
predictions_train_1 = np.reshape(predictions_train_1,(-1,1))
predictions_train_2 = np.reshape(predictions_train_2,(-1,1))
residuals_1 = y_train_1 - predictions_train_1
residuals_2 = y_train_2 - predictions_train_2

# Get the errors.
print("-"*200)
print("The mean absolute error of the test set is %0.2f" % np.average(abs(residuals_1)))
print("The mean squared error of the test set is %0.2f" % np.average(abs(residuals_1)**2))
print("The root mean squared error of the test set is %0.2f" % np.sqrt(np.mean(abs(residuals_1)**2)))
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
ax[0].plot(y_train_1, label='Actual Values')
#y_values_dates = create_dates(X_train_2[-48*7:], predictions_train_2[-48*7:])
ax[0].plot(predictions_train_1, label='Predictions')
ax[0].set_xlabel('Settlement Period Train Set 1')
ax[0].set_ylabel('Electricity Load [MW]')
ax[0].legend(loc="lower right")

#y_values_dates = create_dates(X_train_2[-48*7:],residuals)
ax[1].plot(residuals_1, color = 'black', label='Residuals')
ax[1].axhline(0, linestyle='--', color='k')
ax[1].set_xlabel('Settlement Period Train Set 1')
ax[1].set_ylabel('Electricity Load [MW]')
ax[1].legend(loc="lower right")
plt.show()

########################################################################################################################
# Plot the prediction on the train set 2.
########################################################################################################################

# Get the errors.
print("-"*200)
print("The mean absolute error of the train set 2 is %0.2f" % np.average(abs(residuals_2)))
print("The mean squared error of the train set 2 is %0.2f" % np.average(abs(residuals_2)**2))
print("The root mean squared error of the train set 2 is %0.2f" % np.sqrt(np.mean(abs(residuals_2)**2)))
print("The mean absolute percent error of the train set 2 is %0.2f" % np.mean(abs((y_train_2-predictions_train_2)/y_train_2)))
print("-"*200)

fig, ax = plt.subplots(2)
fig.suptitle('SARIMA Model', fontsize=20)
#y_values_dates = create_dates(X_train_2[-48*7:], y_train_2[-48*7:])
#ax[0].plot(y_values_dates, label='Train Values')
#y_values_dates = create_dates(X_train_2[-48*7:], y_train_2[-48*7:])
ax[0].plot(y_train_2, label='Actual Values')
#y_values_dates = create_dates(X_train_2[-48*7:], predictions_train_2[-48*7:])
ax[0].plot(predictions_train_2, label='Predictions')
ax[0].set_xlabel('Settlement Period Train Set 1')
ax[0].set_ylabel('Electricity Load [MW]')
ax[0].legend(loc="lower right")

#y_values_dates = create_dates(X_train_2[-48*7:],residuals)
ax[1].plot(residuals_2, color = 'black', label='Residuals')
ax[1].axhline(0, linestyle='--', color='k')
ax[1].set_xlabel('Settlement Period Train Set 1')
ax[1].set_ylabel('Electricity Load [MW]')
ax[1].legend(loc="lower right")
plt.show()

########################################################################################################################
# Plot info on the seasonality.
########################################################################################################################
#
# one = np.array(predictions_train[:48*14].reset_index())
# two = np.array(predictions_train[48:48*15].reset_index())
# diff = abs(one[:,1]-two[:,1])
#
# fig, ax = plt.subplots(2)
# fig.suptitle('Difference in seasonality', fontsize=20)
# ax[0].plot(one[:,1], label='First 24 hrs.')
# ax[0].plot(two[:,1], label='2nd 24 hrs.')
# ax[0].set_xlabel('Settlement Period')
# ax[0].set_ylabel('Electricity Load [MW]')
# ax[0].legend(loc="lower right")
# plt.show()
#
# ax[1].plot(diff, color = 'black', label='Difference')
# ax[1].set_xlabel('Settlement Period')
# ax[1].set_ylabel('Electricity Load [MW]')
# ax[1].legend(loc="lower right")
# plt.show()
#
# # residuals.plot_diagnostics(figsize=(7,5))
# # plt.show()

########################################################################################################################
# Save the results in a csv file.
########################################################################################################################

pd.DataFrame(predictions_train_2).to_csv(
    "C:\Python\Pycharm\MSc_Thesis/Electricity_generation/Hybrid_Model/Pred_train2_other_metrics/SARIMA_prediction.csv")
pd.DataFrame(predictions_train_1).to_csv(
    "C:\Python\Pycharm\MSc_Thesis/Electricity_generation/SARIMA/SARIMA_prediction_train_1.csv")

# pd.DataFrame(result_test).to_csv(
#     "/Users/benoitputzeys/PycharmProjects/MSc_Thesis/Electricity_generation/Hybrid_Model/Pred_test_other_metrics/SARIMA_prediction.csv")
#
# import csv
# with open('/Users/benoitputzeys/PycharmProjects/MSc_Thesis/Compare_Models/MST2_results/SARIMA_result.csv', 'w', newline='',) as file:
#     writer = csv.writer(file)
#     writer.writerow(["Method","MSE","MAE","RMSE"])
#     writer.writerow(["SARIMA",
#                      str(mean_squared_error(y_test,result_test)),
#                      str(mean_absolute_error(y_test,result_test)),
#                      str(np.sqrt(mean_squared_error(y_test,result_test)))
#                      ])