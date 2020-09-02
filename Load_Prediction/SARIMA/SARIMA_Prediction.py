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
import matplotlib.ticker as plticker
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error, mean_squared_error


########################################################################################################################
# Importing the data.
########################################################################################################################

# Get the X (containing the features) and y (containing the labels) values
X = pd.read_csv('Data_Preprocessing/For_336_SP_Step_Prediction/X.csv', delimiter=',')
DoW = X["Day of Week"]
X = X.set_index("Time")
dates = X.iloc[:,-1]
# Get rid of unnecessary input features.
X = X.iloc[:,:-6]
y = pd.read_csv('Data_Preprocessing/For_336_SP_Step_Prediction/y.csv', delimiter=',')
y = y.set_index("Time")

# Split data into 80% training set and 20% test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = False)

# Only take half the training and test set (see thesis why).
X_train = X_train[int(len(X_train)*1/2):]
y_train = y_train[int(len(y_train)*1/2):]
dates = dates[-len(X_train)-len(X_test):]
dates_train = dates[:len(X_train)]
dates_test = dates[-len(X_test):]

########################################################################################################################
# Check for stationarity entails differentiation. Plot the difference
########################################################################################################################

y_1 = y_train[1:]
y_2 = y_train[:-1]
# Take the difference between the actual and the previous values.
y_diff = y_1.values - y_2.values

# Plot the difference
plt.figure()
plt.plot(y_diff, linewidth=0.5, color = "blue")
plt.title('Difference Electricity Load SARIMA Model', fontsize=14)
plt.ylabel('Electricity Load, MW', fontsize=14)
plt.grid(True)
plt.show()

########################################################################################################################
# Check the auto-correlation and the partial auto-correlation function.
########################################################################################################################

# Adfuller test
result = adfuller(y_diff)
print("ADF statistic: %f" % result[0])

# Print the p-value.
# If this value is very close to 0, it means it is a positive test in the sense that the series is stationary.
print("P-value: %f" % result[1])
print("Critical values:")

for key, value in result[4].items():
    print("\t%s: %.3f" % (key, value))

# Plot the acf and pacf graphs.
# Anything within the blue lines is not statistically significant. The blue regions are the error bands.
# For ACF, we expect a diminishing trend over time.
fig, ax = plt.subplots(2, figsize=(12,6))
ax[0] = plot_acf(y_diff,ax=ax[0], lags = 50)
ax[1] = plot_pacf(y_diff,ax=ax[1], lags = 50)
plt.show()

########################################################################################################################
# Set the hyperparameters and fit the model only on last 2 weeks because it takes a LONG time.
########################################################################################################################

my_order = (0,1,1)
my_seasonal_order = (0, 1, 1, 48)
model = SARIMAX(y_train[-48*7*2:], order = my_order, seasonal_order = my_seasonal_order)
#model = SARIMAX(y_train_1, order=my_order, seasonal_order=my_seasonal_order)
model_fit = model.fit()

# Summary of the model
print(model_fit.summary())

########################################################################################################################
# Decompose the data into seasonal component, trend and residual error of the 2.
########################################################################################################################

# Decompose the data using an additive model.
ts_decompose = sm.tsa.seasonal_decompose(y_train[-48*7*2:].values, model='additive', period = 48)
ts_decompose.plot()
plt.show()

# Make predictions on the training set and calculate the errors.
predictions_train = model_fit.predict(start = 1, end = 48*7*2)
predictions_train = np.array(predictions_train).reshape(-1,1)
error_train = y_train[-48*7*2:] - predictions_train

# Make predictions on the test set and calculate the errors.
predictions_test = model_fit.predict(start = 48*7*2, end = 48*7*2+48*7-1)
predictions_test = np.array(predictions_test).reshape(-1,1)
error_test = y_test[:48*7] - predictions_test

# Print the errors.
print("-"*200)
print("The mean absolute error of the train set is %0.2f" % np.average(abs(error_train)),"MW")
print("The mean squared error of the train set is %0.2f" % np.average(abs(error_train)**2),"MW^2")
print("The root mean squared error of the train  set is %0.2f" % np.sqrt(np.mean(abs(error_train)**2)),"MW")
print("The mean absolute percent error of the train set is %0.2f" % np.mean(abs((y_train[-48*7*2:] -predictions_train)/y_train)),"%")
print("-"*200)

print("The mean absolute error of the test set is %0.2f" % np.average(abs(error_test)),"MW")
print("The mean squared error of the test set is %0.2f" % np.average(abs(error_test)**2),"MW^2")
print("The root mean squared error of the test set is %0.2f" % np.sqrt(np.mean(abs(error_test)**2)),"MW")
print("The mean absolute percent error of the test set is %0.2f" % np.mean(abs((y_test[:48*7]-predictions_test)/y_test)),"%")
print("-"*200)

########################################################################################################################
# Plot the prediction on train set and test set.
########################################################################################################################

# Plot for the training set.
fig1, ax1 = plt.subplots(2,1,figsize=(12,6))
ax1[0].plot(y_train[-48*7*2:] , label='Train Set', color = "blue")
ax1[0].plot(predictions_train, label='SARIMA predictions', color = "orange")
ax1[0].set_ylabel('Electricity Load, MW', size = 14)
ax1[0].plot(y_train[-1:], label = "Error", color = "red")

ax1[1].plot(error_train, color = 'red')
ax1[1].set_xlabel('Date', size = 14)
ax1[1].set_ylabel('Load, MW', size = 14)

# Include additional details such as grid on and rotation of xaxis...
ax1[0].legend()
ax1[0].grid(True), ax1[1].grid(True)
loc = plticker.MaxNLocator() # this locator puts ticks at regular intervals
ax1[0].xaxis.set_major_locator(loc)
ax1[1].xaxis.set_major_locator(loc)
fig1.autofmt_xdate(rotation = 8)
ax1[0].legend(loc=(1.02,0.7))

fig1.show()

# Plot for the test set.
fig2, ax2 = plt.subplots(2,1,figsize=(12,6))
ax2[0].plot(y_test[:48*7], label='Test Set', color = "black")
ax2[0].plot(predictions_test, label='SARIMA predictions', color = "orange")
ax2[0].set_ylabel('Electricity Load, MW', size = 14)
ax2[0].plot(y_test[-1:], label = "Error", color = "red")

ax2[1].plot(error_test, color = 'red')
ax2[1].set_xlabel('Date', size = 14)
ax2[1].set_ylabel('Load, MW', size = 14)
ax2[1].legend(loc="lower right")

# Include additional details.
ax2[0].legend()
ax2[0].grid(True), ax2[1].grid(True)
loc = plticker.MaxNLocator() # Puts ticks at regular intervals
ax2[0].xaxis.set_major_locator(loc)
ax2[1].xaxis.set_major_locator(loc)
fig2.autofmt_xdate(rotation = 8)
ax2[0].legend(loc=(1.02,0.7))
fig2.show()

########################################################################################################################
# Save the results in a csv file.
########################################################################################################################
#
# pd.DataFrame(predictions_test).to_csv("Load_Prediction/Hybrid_Model/Pred_test_other_metrics/SARIMA_prediction.csv")
#
#df_errors = pd.DataFrame({"MSE_Train": [mean_squared_error(y_train,predictions_train)],
#                          "MAE_Train": [mean_absolute_error(y_train,predictions_train)],
#                          "RMSE_Train": [np.sqrt(mean_squared_error(y_train,predictions_train))],
#                          "MSE_Test": [mean_squared_error(y_test, predictions_test)],
#                          "MAE_Test": [mean_absolute_error(y_test, predictions_test)],
#                          "RMSE_Test": [np.sqrt(mean_squared_error(y_test, predictions_test))],
#                          })
#df_errors.to_csv("Compare_Models/Direct_Multi_Step_Results/SARIMA.csv")