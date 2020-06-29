
import numpy as np
from scipy.ndimage.interpolation import shift
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from pandas import DataFrame
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")

def create_dates(features_df, y_values):

    date_list = [datetime.datetime(year=int(features_df[i, -1]),
                                   month=int(features_df[i, -2]),
                                   day=int(features_df[i, -3]),
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
X = genfromtxt('/Users/benoitputzeys/PycharmProjects/MSc_Thesis/Data_Entsoe/Data_Preprocessing/X.csv', delimiter=',')
y = genfromtxt('/Users/benoitputzeys/PycharmProjects/MSc_Thesis/Data_Entsoe/Data_Preprocessing/y.csv', delimiter=',')

# Set the length of the training data ltd:
ltrd = 48*35
lted = 48*35
X_train = X[-ltrd*2:-lted]
#exogenous_variable = X_train[:,5:9]
X_test = X[-lted:]
y_train = y[-ltrd*2:-lted]
y_test = y[-lted:]

# Plot the first 7 days in the training set.
y_values_dates = create_dates(X_train[:7*48], y_train[:7*48])
plt.figure()
plt.plot(y_values_dates, linewidth=0.5)
plt.title('Electricity Load SARIMA Model', fontsize=20)
plt.ylabel('Electricity Load [MW]', fontsize=16)

########################################################################################################################
# Check for stationarity
########################################################################################################################

y_1 = shift(y_train, 1, cval=np.NaN)
y_2 = y_train
y_1 = np.delete(y_1,0)
y_2 = np.delete(y_2,0)

# Take the difference between the actual and the previous values.
y_diff = y_1 - y_2

# Plot the difference.
y_values_dates = create_dates(X_train[:48*7], y_diff[:48*7])
plt.figure()
plt.plot(y_values_dates, linewidth=0.5)
plt.title('Electricity Load SARIMA Model', fontsize=20)
plt.ylabel('Electricity Load [MW]', fontsize=16)

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

########################################################################################################################
# Set the hyperparameters and fit the model.
########################################################################################################################

my_order = (1,1,3)
my_seasonal_order = (1, 0, 1, 48)
#model = SARIMAX(y_train, order=my_order, seasonal_order=my_seasonal_order, exog=exogenous_variable)
model = SARIMAX(y_train, order=my_order, seasonal_order=my_seasonal_order)
model_fit = model.fit()

# Summary of the model
print(model_fit.summary())

########################################################################################################################
# Decompose the data into seasonal component, trend and residual error of the 2.
########################################################################################################################

# Decompose the data.
ts_decompose = sm.tsa.seasonal_decompose(y_train[0:48*5], model='additive', period = 48)
ts_decompose.plot()
plt.show

# Get the prediction and its residual.
# Define the lenght of the prediciton.
lprd = 48*7
predictions = model_fit.predict(lprd, exog=X_test[:lprd,1])
predictions = pd.Series(predictions)
residuals = y_test[:lprd] - predictions[:lprd]

# Get the errors.
print("-"*200)
print("The mean absolute error of the test set is %0.2f" % np.average(abs(residuals)))
print("The mean squared error of the test set is %0.2f" % np.average(abs(residuals)**2))
print("The root mean squared error of the test set is %0.2f" % np.sqrt(np.mean(abs(residuals)**2)))
print("The mean absolute percent error of the test set is %0.2f" % np.mean(abs((y_test[:lprd]-predictions[:lprd])/y_test[:lprd])))
print("-"*200)

########################################################################################################################
# Plot the prediction.
########################################################################################################################

fig, ax = plt.subplots(2)
fig.suptitle('SARIMA Model', fontsize=20)
y_values_dates = create_dates(X_train[-48*7:], y_train[-48*7:])
ax[0].plot(y_values_dates, label='Train Values')
y_values_dates = create_dates(X_test[:lprd],y_test[:lprd])
ax[0].plot(y_values_dates, label='Actual Values')
y_values_dates = create_dates(X_test[:lprd],np.array(predictions[:lprd]))
ax[0].plot(y_values_dates, label='Predictions')
ax[0].set_xlabel('Settlement Period')
ax[0].set_ylabel('Electricity Load [MW]')
ax[0].legend(loc="lower right")
plt.show()

y_values_dates = create_dates(X_test[:48*3],np.array(residuals))
ax[1].plot(y_values_dates, color = 'black', label='Residuals')
ax[1].axhline(0, linestyle='--', color='k')
ax[1].set_xlabel('Settlement Period')
ax[1].set_ylabel('Electricity Load [MW]')
ax[1].legend(loc="lower right")
plt.show()

########################################################################################################################
# Plot info on the seasonality.
########################################################################################################################

one = np.array(predictions[:48*14].reset_index())
two = np.array(predictions[48:48*15].reset_index())
diff = abs(one[:,1]-two[:,1])

fig, ax = plt.subplots(2)
fig.suptitle('Difference in seasonality', fontsize=20)
ax[0].plot(one[:,1], label='First 24 hrs.')
ax[0].plot(two[:,1], label='2nd 24 hrs.')
ax[0].set_xlabel('Settlement Period')
ax[0].set_ylabel('Electricity Load [MW]')
ax[0].legend(loc="lower right")
plt.show()

ax[1].plot(diff, color = 'black', label='Difference')
ax[1].set_xlabel('Settlement Period')
ax[1].set_ylabel('Electricity Load [MW]')
ax[1].legend(loc="lower right")
plt.show()

# residuals.plot_diagnostics(figsize=(7,5))
# plt.show()

########################################################################################################################
# Save the results in a csv file.
########################################################################################################################

pd.DataFrame(predictions[:48*7]).to_csv("/Users/benoitputzeys/PycharmProjects/MSc_Thesis/Electricity_generation/Hybrid_Model/SARIMA_prediction.csv")

