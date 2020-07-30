from sklearn.svm import SVR
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
from pandas import DataFrame
import matplotlib.ticker as plticker

########################################################################################################################
# Get data and data preprocessing.
########################################################################################################################

# Get the X (containing the features) and y (containing the labels) values
X = pd.read_csv('Data_Preprocessing/For_Multi_Step_Prediction/X.csv', delimiter=',')
X = X.set_index("Time")
dates = X.iloc[:,-1]
X = X.iloc[:,:-6]

y = pd.read_csv('Data_Preprocessing/For_Multi_Step_Prediction/y.csv', delimiter=',')
y = y.set_index("Time")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = False)

X_train = X_train[int(len(X_train)*1/2):]
X_test = X_test[:int(len(X_test)*1/2)]
y_train = y_train[int(len(y_train)*1/2):]
y_test = y_test[:int(len(y_test)*1/2)]
dates = dates[-len(X_train)-len(X_test)*2:-len(X_test)]

X_train_unscaled = X_train
X_test_unscaled = X_test

# Feature Scaling
x_scaler = StandardScaler()
y_scaler = StandardScaler()
X_train = x_scaler.fit_transform(X_train)
X_test = x_scaler.transform(X_test)
y_train = y_scaler.fit_transform(y_train)

########################################################################################################################
# Create the model.
########################################################################################################################

# Fit the SVR to our data
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)

########################################################################################################################
# Predicting the generation on the test set and inverse the scaling.
########################################################################################################################

pred_train = y_scaler.inverse_transform(regressor.predict(X_train))/1000
pred_test = y_scaler.inverse_transform(regressor.predict(X_test))/1000

X_train = x_scaler.inverse_transform(X_train)
X_train[:,0] = X_train[:,0]/1000
X_test = x_scaler.inverse_transform(X_test)
X_test[:,0] = X_test[:,0]/1000
y_train = (y_scaler.inverse_transform(y_train)/1000).reshape(-1,)
y_test = np.array(y_test.iloc[:,-1]/1000)

########################################################################################################################
# Compute and print the errors.
########################################################################################################################

# Compute the error between the Actual Generation and the prediction from the NN
print("-"*200)
error_train = (pred_train - y_train)
print("The mean absolute error of the train set is %0.2f" % mean_absolute_error(y_train,pred_train))
print("The mean squared error of the train set is %0.2f" % mean_squared_error(y_train,pred_train))
print("The root mean squared error of the train set is %0.2f" % np.sqrt(mean_squared_error(y_train,pred_train)))
print("-"*200)

error_test = (pred_test - y_test)
print("The mean absolute error of the test set is %0.2f" % mean_absolute_error(y_test,pred_test))
print("The mean squared error of the test set is %0.2f" % mean_squared_error(y_test,pred_test))
print("The root mean squared error of the test set is %0.2f" % np.sqrt(mean_squared_error(y_test,pred_test)))
print("-"*200)

########################################################################################################################
# Visualising the results
########################################################################################################################

error_test_plot = np.zeros((48*3+48*7,1))
error_test_plot[-336:] = error_test[:48*7].reshape(-1,1)

# Plot the result with the truth in red and the predictions in blue.
fig2, axs2=plt.subplots(2,1,figsize=(12,6))

# First plot contains the prediction and the true values from the test and training set.
axs2[0].plot(dates.iloc[-len(X_test)-48*3:-len(X_test)],
             y_train[-48*3:],
             label = "Training Set (True Values)", alpha = 1, color = "blue")
axs2[0].plot(dates.iloc[-len(X_test):-len(X_test)+48*7],
             pred_test[:48*7],
             label = "SVR Pred.", color = "orange")
axs2[0].plot(dates.iloc[-len(X_test):-len(X_test)+48*7],
             y_test[:48*7],
             label = "Test Set (True Values)", alpha = 1, color = "black")
axs2[0].axvline(dates.iloc[-len(X_test)], linestyle="--", color = "black")
axs2[0].set_ylabel('Load [GW]',size = 14)

# Second plot contains the errors.
axs2[1].plot(dates.iloc[-len(X_test)-48*3:-len(X_test)+48*7],
             error_test_plot,
             label = "Error", alpha = 1, color = "red")
axs2[1].axvline(dates.iloc[-len(X_test)], linestyle="--", color = "black")

# Include additional details such as tick intervals, rotation, legend positioning and grid on.
axs2[0].grid(True)
axs2[1].grid(True)
axs2[1].set_xlabel('Date',size = 14)
axs2[1].set_ylabel('Error [GW]',size = 14)
loc = plticker.MultipleLocator(base=47) # this locator puts ticks at regular intervals
axs2[1].xaxis.set_major_locator(loc)
axs2[0].xaxis.set_major_locator(loc)
fig2.autofmt_xdate(rotation=12)
axs2[1].legend(loc=(1.04,0.9))
axs2[0].legend(loc=(1.04,0.7))
fig2.show()

########################################################################################################################
# Compute the standard deviation of the training set.
########################################################################################################################

X = pd.read_csv('Data_Preprocessing/For_Multi_Step_Prediction/X.csv', delimiter=',')
settlement_period_week = X["Settlement Period"]+(48*X["Day of Week"])

dates_train = dates.iloc[:len(X_train)]
dates_test = dates.iloc[-len(X_test):]
train_set = y_train
settlement_period_train = settlement_period_week[-len(X_test)*2-len(X_train):-len(X_test)*2]

# Create a dataframe that contains the SPs (1-336) and the load values.
error_train = pd.DataFrame({'SP':settlement_period_train, 'Error_Train': (pred_train-train_set)})

# Plot the projected errors onto a single week to see the variation in the timeseries.
fig3, axs3=plt.subplots(1,1,figsize=(12,6))
axs3.scatter(error_train["SP"],
             error_train["Error_Train"],
             alpha=0.05, label = "Projected Errors", color = "red")
axs3.set_ylabel("Error during training [GW]", size = 14)
axs3.set_xlabel("Settlement Period", size = 14)
axs3.grid(True)
axs3.legend()
fig3.show()

# Compute the mean and variation for each x.
training_stats = pd.DataFrame({'Index':np.linspace(1,336,336), 'Mean':np.linspace(1,336,336), 'Stddev':np.linspace(1,336,336)})

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
axs4.set_ylabel("Error during training [GW]", size = 14)
axs4.set_xlabel("Settlement Period / Weekday", size = 14)

# Include additional details such as tick intervals, legend positioning and grid on.
loc = plticker.MultipleLocator(base=47)
plt.xticks(np.arange(1,385, 48), ["1 / Monday", "49 / Tuesday", "97 / Wednesday", "145 / Thursday", "193 / Friday","241 / Saturday", "289 / Sunday",""])
axs4.legend()
axs4.grid(True)
fig4.show()

stddev = training_stats["Stddev"]

fig5, axs5=plt.subplots(2,1,figsize=(12,6))
# First plot contains the prediction, the true values from the test and training set and the standard deviation.
axs5[0].plot(dates.iloc[-len(X_test)-48*3:-len(X_test)],
             y_train[-48*3:],
             label = "Training Set (True Values)", alpha = 1, color = "blue")
axs5[0].plot(dates.iloc[-len(X_test):-len(X_test)+48*7],
             pred_test[:48*7],
             label = "SVR Pred.", color = "orange")
axs5[0].plot(dates.iloc[-len(X_test):-len(X_test)+48*7],
             y_test[:48*7],
             label = "Test Set (True Values)", alpha = 1, color = "black")
axs5[0].fill_between(dates.iloc[-len(X_test)+29:-len(X_test)+48*7],
                    pred_test[29:48*7]+stddev[29:],
                    pred_test[29:48*7]-stddev[29:],
                    label = "+-1 x Standard Deviation", alpha = 0.2, color = "orange")
axs5[0].fill_between(dates.iloc[-len(X_test):-len(X_test)+29],
                    pred_test[:29]+stddev[:29],
                    pred_test[:29]-stddev[:29],
                    alpha = 0.2, color = "orange")
axs5[0].axvline(dates.iloc[-len(X_test)], linestyle="--", color = "black")
axs5[0].set_ylabel('Load [GW]',size = 14)

# Second plot contains the errors.
axs5[1].plot(dates.iloc[-len(X_test)-48*3:-len(X_test)+48*7],
             error_test_plot,
             label = "Error", alpha = 1, color = "red")
axs5[1].axvline(dates.iloc[-len(X_test)],
                linestyle="--", color = "black")
axs5[1].set_xlabel('Date',size = 14)
axs5[1].set_ylabel('Error [GW]',size = 14)

# Include additional details such as tick intervals, rotation, legend positioning and grid on.
axs5[1].grid(True)
axs5[0].grid(True)
loc = plticker.MultipleLocator(base=47) # Put ticks at regular intervals
axs5[0].xaxis.set_major_locator(loc)
axs5[1].xaxis.set_major_locator(loc)
fig5.autofmt_xdate(rotation=15)
axs5[1].legend(loc=(1.04,0.9))
axs5[0].legend(loc=(1.04,0.6))

fig5.show()

########################################################################################################################
# Save the results in a csv file.
########################################################################################################################

import csv
with open('Compare_Models/Single_Multi_Step_results/SVR.csv', 'w', newline='', ) as file:
    writer = csv.writer(file)
    writer.writerow(["Method","MSE","MAE","RMSE"])
    writer.writerow(["SVR",
                     str(mean_squared_error(y_test,pred_test)),
                     str(mean_absolute_error(y_test,pred_test)),
                     str(np.sqrt(mean_squared_error(y_test,pred_test)))
                     ])

import csv
with open('Compare_Models/SMST_Probability_results/Probability_Based_on_Training/SVR_error.csv', 'w', newline='', ) as file:
    writer = csv.writer(file)
    writer.writerow(["Method","MSE","MAE","RMSE"])
    writer.writerow(["SVR",
                     str(mean_squared_error(y_test,pred_test)),
                     str(mean_absolute_error(y_test,pred_test)),
                     str(np.sqrt(mean_squared_error(y_test,pred_test)))
                     ])

training_stats.to_csv("Compare_Models/SMST_Probability_results/Probability_Based_on_Training/SVR_mean_errors_stddevs.csv")
