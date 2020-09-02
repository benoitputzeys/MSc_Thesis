import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.ticker as plticker
import keras
import datetime
import matplotlib.dates as mdates

########################################################################################################################
# Get data and data preprocessing.
########################################################################################################################

# Get the X (containing the features) and y (containing the labels) values
X = pd.read_csv('Data_Preprocessing/For_336_SP_Step_Prediction/X.csv', delimiter=',')
X = X.set_index("Time")
dates = X.iloc[:,-1]
X = X.iloc[:,:-6]
y = pd.read_csv('Data_Preprocessing/For_336_SP_Step_Prediction/y.csv', delimiter=',')
y = y.set_index("Time")

# Divide the data into 80% training and 20% test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = False)

# Only include half the length of the training data and half the test set (according to the results found in the thesis.
# Divide by 1000 to express everythin in GW.
X_train = X_train[int(len(X_train)*1/2):]/1000
X_test = X_test[:int(len(X_test)*1/2)]/1000
y_train = y_train[int(len(y_train)*1/2):]/1000
y_test = y_test[:int(len(y_test)*1/2)]/1000
dates = dates[-len(X_train)-len(X_test)*2:-len(X_test)]

########################################################################################################################
# Import the predicitons from the LSTM
########################################################################################################################

pred_test = pd.read_csv('Load_Prediction/LSTM/Direct_Multi_Step_Prediction/Pred_Test.csv', delimiter=',')
pred_train = pd.read_csv('Load_Prediction/LSTM/Direct_Multi_Step_Prediction/Pred_Train.csv', delimiter=',')
pred_train = pred_train.iloc[:,-1]
pred_test = pred_test.iloc[:,-1]

########################################################################################################################
# Compute the standard deviation of the errors from the training set.
# Contains 2 graphs, 1. the scatter plot from the errors during training and 2. the actual confidence band derived from
# the scatter plot.
########################################################################################################################

# Load the data with the SP still as input feature.
X = pd.read_csv('Data_Preprocessing/For_336_SP_Step_Prediction/X.csv', delimiter=',')

# Create settlement periods for a week going from 1 to 336.
settlement_period_week = X["Settlement Period"]+(48*X["Day of Week"])

# Make sure the dates correspond to the values in the training and testing set.
dates_train = dates.iloc[:len(X_train)]
dates_test = dates.iloc[-len(X_test):]

# Include the settlement periods specific to the training set.
settlement_period_train = settlement_period_week[-len(X_test)*2-len(X_train):-len(X_test)*2]

# Create a dataframe that contains the SPs (1-336) and the load values.
error_train = pd.DataFrame({'SP':settlement_period_train, 'Error_Train': (pred_train.values-y_train.values.reshape(-1,))})

# Plot the projected errors onto a single week to see the variation in the timeseries.
fig3, axs3=plt.subplots(1,1,figsize=(12,6))
axs3.scatter(error_train["SP"],
             error_train["Error_Train"],
             alpha=0.05, label = "Projected Errors", color = "red")
axs3.set_ylabel("LSTM error during training, GW", size = 14)
axs3.set_xlabel("Settlement Period", size = 14)
axs3.grid(True)
axs3.legend()
fig3.show()

# Compute the mean and variation for each SP.
training_stats = pd.DataFrame({'Index':np.linspace(1,336,336),
                               'Mean':np.linspace(1,336,336),
                               'Stddev':np.linspace(1,336,336)})

for i in range(1,337):
    training_stats.iloc[i-1,1]=np.mean(error_train[error_train["SP"]==i].iloc[:,-1])
    training_stats.iloc[i-1,2]=np.std(error_train[error_train["SP"]==i].iloc[:,-1])

# Plot the mean and standard deviation of the errors that are made on the training set in orange.
fig4, axs4=plt.subplots(1,1,figsize=(12,6))
axs4.plot(training_stats.iloc[:,0],
          training_stats.iloc[:,1],
          color = "orange", label = "Mean of all projected errors")
axs4.fill_between(training_stats.iloc[:,0],
                  (training_stats.iloc[:,1]-training_stats.iloc[:,2]),
                  (training_stats.iloc[:,1]+training_stats.iloc[:,2]),
                  alpha=0.2, color = "orange", label = "+- 1x Standard Deviation")
axs4.set_ylabel("LSTM error during training, GW", size = 14)

# Include additional details such as tick intervals, legend positioning and grid on.
axs4.set_xticks(np.arange(1,385, 24))
axs4.set_xticklabels(["00:00\nMonday","12:00",
                       "00:00\nTuesday","12:00",
                       "00:00\nWednesday", "12:00",
                       "00:00\nThursday", "12:00",
                       "00:00\nFriday","12:00",
                       "00:00\nSaturday", "12:00",
                       "00:00\nSunday","12:00",
                       "00:00"])
axs4.grid(b=True, which='major'), axs4.grid(b=True, which='minor',alpha = 0.2)
axs4.tick_params(axis = "both", labelsize = 12)
axs4.minorticks_on()
axs4.legend(fontsize=14)
axs4.set_axisbelow(True)
fig4.show()
# Save the figure
fig4.savefig("Load_Prediction/LSTM/Figures/DMSP_Mean_and_Stddev_of_Error_Train_Set_Pred.pdf", bbox_inches='tight')

########################################################################################################################
# Make the prediction with the orange band, the confidence interval.
########################################################################################################################

# Extract the standard deviation from the errors during the training set.
stddev = training_stats["Stddev"]

# Prepare a column vector that contains the error between the test set values and the prediction.
error_test = pred_test.values - y_test.values.reshape(-1,)
error_test_plot = np.zeros((48*3+48*7,1))
error_test_plot[-336:] = error_test[:48*7].reshape(-1,1)

fig5, axs5=plt.subplots(2,1,figsize=(12,6))
# First plot contains the prediction (orange), the test set values (black) and training set values (blue)
# and the standard deviation (orange).
axs5[0].plot(dates.iloc[-len(X_test)-48*3:-len(X_test)],
             y_train[-48*3:],
             label = "Training Set", alpha = 1, color = "blue")
axs5[0].plot(dates.iloc[-len(X_test):-len(X_test)+48*7],
             pred_test[:48*7],
             label = "LSTM Pred.", color = "orange")
axs5[0].plot(dates.iloc[-len(X_test):-len(X_test)+48*7],
             y_test[:48*7],
             label = "Test Set", alpha = 1, color = "black")

# Use the blue band from Thursday 14:00 to Sunday 23:30 (start at SP 173)
axs5[0].fill_between(dates.iloc[-len(X_test):-len(X_test)+163],
                    pred_test[:163].values+stddev[173:].values,
                    pred_test[:163].values-stddev[173:].values,
                    alpha = 0.2, color = "orange")

# Use the blue band from Monday 00:00 (SP = 1) to Thursday 13:30 (SP=173)
axs5[0].fill_between(dates.iloc[-len(X_test)+163:-len(X_test)+48*7],
                    pred_test[163:48*7].values+stddev[:173].values,
                    pred_test[163:48*7].values-stddev[:173].values,
                    label = "+-1 x\nStandard Deviation", alpha = 0.2, color = "orange")

axs5[0].axvline(dates.iloc[-len(X_test)], linestyle="--", color = "black")
axs5[0].set_ylabel('Load, GW',size = 14)
axs5[0].plot(30,30,label = "Error", color = "red")

# Second plot contains the errors.
axs5[1].plot(dates.iloc[-len(X_test)-48*3:-len(X_test)+48*7],
             error_test_plot,
             label = "Error", alpha = 1, color = "red")
axs5[1].axvline(dates.iloc[-len(X_test)],
                linestyle="--", color = "black")
axs5[1].set_xlabel('Date',size = 14)
axs5[1].set_ylabel('Error, GW',size = 14)

# Include additional details such as tick intervals, rotation, legend positioning and grid on.
axs5[1].grid(True)
axs5[0].grid(True)
loc = plticker.MultipleLocator(base=48)
axs5[0].xaxis.set_major_locator(loc) # Put ticks at regular intervals
axs5[1].xaxis.set_major_locator(loc)
axs5[0].set_axisbelow(True), axs5[0].set_axisbelow(True)
fig5.autofmt_xdate(rotation=0)
axs5[0].legend(loc=(1.02,0.48))
plt.xticks(np.arange(1,482, 48), ["14:00\n07/22","14:00\n07/23","14:00\n07/24",
                                  "14:00\n07/25","14:00\n07/26","14:00\n07/27",
                                  "14:00\n07/28","14:00\n07/29","14:00\n07/30",
                                  "14:00\n07/31","14:00\n08/01"])
fig5.show()
# Save the figure.
fig5.savefig("Load_Prediction/LSTM/Figures/DMSP_Pred_w_Uncertainty.pdf", bbox_inches='tight')

########################################################################################################################
# Compute the standard deviation of the errors from the test set.
# Contains 2 graphs, 1. the scatter plot from the errors during testing and 2. the actual confidence band derived from
# the scatter plot.
########################################################################################################################

# Include the settlement periods specific to the test set.
settlement_period_test = settlement_period_week[-len(X_test)*2:-len(X_test)]

# Create a dataframe that contains the SPs (1-336) and the load values.
error_test = pd.DataFrame({'SP':settlement_period_test, 'Error_Test': (pred_test.values-y_test.values.reshape(-1,))})

# Plot the projected errors onto a single week to see the variation in the timeseries.
fig6, axs6=plt.subplots(1,1,figsize=(12,6))
axs6.scatter(error_test["SP"],
             error_test["Error_Test"],
             alpha=0.05, label = "Projected Errors", color = "red")
axs6.set_ylabel("LSTM error during test set, GW", size = 14)
axs6.set_xlabel("Settlement Period", size = 14)
axs6.grid(True)
axs6.legend()
fig6.show()

# Compute the mean and variation for each SP.
test_stats = pd.DataFrame({'Index':np.linspace(1,336,336),
                               'Mean':np.linspace(1,336,336),
                               'Stddev':np.linspace(1,336,336)})

for i in range(1,337):
    test_stats.iloc[i-1,1]=np.mean(error_test[error_test["SP"]==i].iloc[:,-1])
    test_stats.iloc[i-1,2]=np.std(error_test[error_test["SP"]==i].iloc[:,-1])

# Plot the mean and standard deviation of the errors that are made on the test set.
fig7, axs7=plt.subplots(1,1,figsize=(12,6))
axs7.plot(test_stats.iloc[:,0],
          test_stats.iloc[:,1],
          color = "orange", label = "Mean of all projected errors")
axs7.fill_between(test_stats.iloc[:,0],
                  (test_stats.iloc[:,1]-test_stats.iloc[:,2]),
                  (test_stats.iloc[:,1]+test_stats.iloc[:,2]),
                  alpha=0.2, color = "orange", label = "+- 1x Standard Deviation")
axs7.set_ylabel("LSTM error during test set, GW", size = 14)

# Include additional details such as tick intervals, legend positioning and grid on.
axs7.set_xticks(np.arange(1,385, 24))
axs7.set_xticklabels(["00:00\nMonday","12:00",
                       "00:00\nTuesday","12:00",
                       "00:00\nWednesday", "12:00",
                       "00:00\nThursday", "12:00",
                       "00:00\nFriday","12:00",
                       "00:00\nSaturday", "12:00",
                       "00:00\nSunday","12:00",
                       "00:00"])
axs7.grid(b=True, which='major'), axs7.grid(b=True, which='minor',alpha = 0.2)
axs7.tick_params(axis = "both", labelsize = 12)
axs7.minorticks_on()
axs7.set_axisbelow(True)
axs7.legend(fontsize=14)
fig7.show()
# Save the figure.
fig7.savefig("Load_Prediction/LSTM/Figures/DMSP_Mean_and_Stddev_of_Error_Test_Set_Pred.pdf", bbox_inches='tight')

########################################################################################################################
# Save the results in a csv file.
########################################################################################################################

training_stats.to_csv("Compare_Models/Direct_Multi_Step_Probability_Results/Probability_Based_on_Training/LSTM_mean_errors_stddevs_train.csv")
test_stats.to_csv("Compare_Models/Direct_Multi_Step_Probability_Results/Probability_Based_on_Training/LSTM_mean_errors_stddevs_test.csv")

########################################################################################################################
# Make a figure to show how the standard deviation is computed.
########################################################################################################################

# Prediction on training set.
fig8, axs8=plt.subplots(3,1,figsize=(12,6))
# First plot contains the prediction, the true values from the test and training set.
axs8[0].plot(dates_train[154:48*7*3+154+1],
             y_train[154:48*7*3+154+1],
             label = "Training Set", alpha = 1, color = "blue")
axs8[0].plot(dates_train[154:48*7*3+154+1],
             pred_train[154:48*7*3+154+1],
             label = "LSTM Pred.", color = "orange")
axs8[0].set_ylabel('Load, GW',size = 14)
axs8[0].plot(30,30,label = "Error", color = "red")
loc1 = plticker.MultipleLocator(base=48*7) # this locator puts ticks at regular intervals
axs8[0].xaxis.set_major_locator(loc1)
axs8[0].set_xticks(np.arange(1,4*336, 336))
axs8[0].set_xticklabels(["2017/10/16",
                       "2017/10/23",
                       "2017/10/30",
                       "2017/11/06",
                       ])
# Second plot contains the errors.
axs8[1].plot(dates_train[154:48*7*3+154+1],
             error_train.iloc[154:48*7*3+154+1,-1],
             alpha = 1, color = "red")
axs8[1].set_ylabel('Error, GW',size = 14)
axs8[1].xaxis.set_major_locator(loc1)
axs8[1].set_xticks(np.arange(1,4*336, 336))
axs8[1].set_xticklabels(["2017/10/16",
                       "2017/10/23",
                       "2017/10/30",
                       "2017/11/06",
                       ])

# # Third plot contains the errors projected on a single week
# # Only plot 3000 errors otherwise the file is too large.
axs8[2].scatter(error_train["SP"].iloc[:3000],
             error_train["Error_Train"].iloc[:3000],linewidth = 0.01,
             alpha=0.2, color = "red")
axs8[2].set_ylabel("Projected Error\nduring training, GW", size = 14)

# Include additional details such as tick intervals, rotation, legend positioning and grid on.
axs8[1].grid(True), axs8[0].grid(True)
loc2 = plticker.MultipleLocator(base=24) # this locator puts ticks at regular intervals
axs8[0].set_axisbelow(True), axs8[1].set_axisbelow(True)
axs8[1].tick_params(axis = "both",labelsize = 12)
axs8[0].tick_params(axis = "both",labelsize = 12)
axs8[0].legend(loc=(1.02,0.75))

fig8.show()
# Save the figure.
fig8.savefig("Load_Prediction/LSTM/Figures/DMSP_Projection_Explained.pdf", bbox_inches='tight')
