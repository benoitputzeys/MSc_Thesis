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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = False)

X_train = X_train[int(len(X_train)*1/2):]/1000
X_test = X_test[:int(len(X_test)*1/2)]/1000
y_train = y_train[int(len(y_train)*1/2):]/1000
y_test = y_test[:int(len(y_test)*1/2)]/1000
dates = dates[-len(X_train)-len(X_test)*2:-len(X_test)]

########################################################################################################################
# Import the predicitons.
########################################################################################################################

pred_test = pd.read_csv('Electricity_Generation_Prediction\ANN\Direct_Multi_Step_Prediction\Pred_Test.csv', delimiter=',')
pred_train = pd.read_csv('Electricity_Generation_Prediction\ANN\Direct_Multi_Step_Prediction\Pred_Train.csv', delimiter=',')
pred_train = pred_train.iloc[:,-1]
pred_test = pred_test.iloc[:,-1]

########################################################################################################################
# Compute the standard deviation of the errors from the training set.
# Contains 2 graphs, the scatter plot and the actual confidence band.
########################################################################################################################

X = pd.read_csv('Data_Preprocessing/For_336_SP_Step_Prediction/X.csv', delimiter=',')
settlement_period_week = X["Settlement Period"]+(48*X["Day of Week"])

dates_train = dates.iloc[:len(X_train)]
dates_test = dates.iloc[-len(X_test):]
settlement_period_train = settlement_period_week[-len(X_test)*2-len(X_train):-len(X_test)*2]

# Create a dataframe that contains the SPs (1-336) and the load values.
error_train = pd.DataFrame({'SP':settlement_period_train, 'Error_Train': (pred_train.values-y_train.values.reshape(-1,))})

# Plot the projected errors onto a single week to see the variation in the timeseries.
fig3, axs3=plt.subplots(1,1,figsize=(12,6))
axs3.scatter(error_train["SP"],
             error_train["Error_Train"],
             alpha=0.05, label = "Projected Errors", color = "red")
axs3.set_ylabel("NN error during training, GW", size = 14)
axs3.set_xlabel("Settlement Period", size = 14)
axs3.grid(True)
axs3.legend()
fig3.show()

# Compute the mean and variation for each x.
training_stats = pd.DataFrame({'Index':np.linspace(1,336,336),
                               'Mean':np.linspace(1,336,336),
                               'Stddev':np.linspace(1,336,336)})

for i in range(1,337):
    training_stats.iloc[i-1,1]=np.mean(error_train[error_train["SP"]==i].iloc[:,-1])
    training_stats.iloc[i-1,2]=np.std(error_train[error_train["SP"]==i].iloc[:,-1])

# Plot the mean and standard deviation of the errors that are made on the training set.
fig4, axs4=plt.subplots(1,1,figsize=(12,6))
axs4.plot(training_stats.iloc[:,0],
          training_stats.iloc[:,1],
          color = "orange", label = "Mean of all projected errors")
axs4.fill_between(training_stats.iloc[:,0],
                  (training_stats.iloc[:,1]-training_stats.iloc[:,2]),
                  (training_stats.iloc[:,1]+training_stats.iloc[:,2]),
                  alpha=0.2, color = "orange", label = "+- 1x Standard Deviation")
axs4.set_ylabel("NN error during training, GW", size = 14)

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
fig4.show()
fig4.savefig("Electricity_Generation_Prediction/ANN/Figures/DMST_Mean_and_Stddev_of_Error_Train_Set_Pred.pdf", bbox_inches='tight')

########################################################################################################################
# Make the prediction with the orange band, the confidence interval.
########################################################################################################################

stddev = training_stats["Stddev"]
error_test = pred_test.values - y_test.values.reshape(-1,)
error_test_plot = np.zeros((48*3+48*7,1))
error_test_plot[-336:] = error_test[:48*7].reshape(-1,1)

fig5, axs5=plt.subplots(2,1,figsize=(12,6))
# First plot contains the prediction, the true values from the test and training set and the standard deviation.
axs5[0].plot(dates.iloc[-len(X_test)-48*3:-len(X_test)],
             y_train[-48*3:],
             label = "Training Set", alpha = 1, color = "blue")
axs5[0].plot(dates.iloc[-len(X_test):-len(X_test)+48*7],
             pred_test[:48*7],
             label = "NN Pred.", color = "orange")
axs5[0].plot(dates.iloc[-len(X_test):-len(X_test)+48*7],
             y_test[:48*7],
             label = "Test Set", alpha = 1, color = "black")
# Use the blue band from Thursday 14:00 to Sunday 23:30 (corresponds to an interval of 164 SPs)
axs5[0].fill_between(dates.iloc[-len(X_test):-len(X_test)+164],
                    pred_test[:164].values+stddev[-164:].values,
                    pred_test[:164].values-stddev[-164:].values,
                    alpha = 0.2, color = "orange")
# Use the blue band from Monday 00:00 (SP = 1) to Thursday 13:30 (SP=164)
axs5[0].fill_between(dates.iloc[-len(X_test)+164:-len(X_test)+48*7],
                    pred_test[164:48*7].values+stddev[:172].values,
                    pred_test[164:48*7].values-stddev[:172].values,
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
loc = plticker.MultipleLocator(base=47)
axs5[0].xaxis.set_major_locator(loc) # Put ticks at regular intervals
axs5[1].xaxis.set_major_locator(loc)
fig5.autofmt_xdate(rotation=0)
axs5[0].legend(loc=(1.02,0.48))
plt.xticks(np.arange(1,482, 48), ["14:00\n07/22","14:00\n07/23","14:00\n07/24",
                                  "14:00\n07/25","14:00\n07/26","14:00\n07/27",
                                  "14:00\n07/28","14:00\n07/29","14:00\n07/30",
                                  "14:00\n07/31","14:00\n08/01"])
fig5.show()
fig5.savefig("Electricity_Generation_Prediction/ANN/Figures/DMST_Pred_w_Uncertainty.pdf", bbox_inches='tight')

########################################################################################################################
# Compute the standard deviation of the errors from the test set.
# Contains 2 graphs, the scatter plot and the actual confidence band.
########################################################################################################################

settlement_period_test = settlement_period_week[-len(X_test)*2:-len(X_test)]

# Create a dataframe that contains the SPs (1-336) and the load values.
error_test = pd.DataFrame({'SP':settlement_period_test, 'Error_Test': (pred_test.values-y_test.values.reshape(-1,))})

# Plot the projected errors onto a single week to see the variation in the timeseries.
fig6, axs6=plt.subplots(1,1,figsize=(12,6))
axs6.scatter(error_test["SP"],
             error_test["Error_Test"],
             alpha=0.05, label = "Projected Errors", color = "red")
axs6.set_ylabel("NN error during test set, GW", size = 14)
axs6.set_xlabel("Settlement Period", size = 14)
axs6.grid(True)
axs6.legend()
fig6.show()

# Compute the mean and variation for each x.
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
axs7.set_ylabel("NN error during test set, GW", size = 14)

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
axs7.grid(b=True, which='major'), axs4.grid(b=True, which='minor',alpha = 0.2)
axs7.tick_params(axis = "both", labelsize = 12)
axs7.minorticks_on()
axs7.legend(fontsize=14)
fig7.show()
fig7.savefig("Electricity_Generation_Prediction/ANN/Figures/DMST_Mean_and_Stddev_of_Error_Test_Set_Pred.pdf", bbox_inches='tight')

########################################################################################################################
# Save the results in a csv file.
########################################################################################################################

training_stats.to_csv("Compare_Models/Direct_Multi_Step_Probability_Results/Probability_Based_on_Training/NN_mean_errors_stddevs_train.csv")
test_stats.to_csv("Compare_Models/Direct_Multi_Step_Probability_Results/Probability_Based_on_Training/NN_mean_errors_stddevs_test.csv")

