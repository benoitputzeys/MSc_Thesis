import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from Electricity_Generation_Prediction.LSTM.Functions_LSTM import plot_the_loss_curve, train_model, create_model, plot_generation, plot_prediction_zoomed_in
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.ticker as plticker
import keras

########################################################################################################################
# Get data and data preprocessing.
########################################################################################################################

# Get the X (containing the features) and y (containing the labels) values
X = pd.read_csv('Data_Preprocessing/For_Multi_Step_Prediction/X.csv', delimiter=',')
X = X.set_index("Time")
dates = X.iloc[:,-1]
X = X.iloc[:,:-5]

y = pd.read_csv('Data_Preprocessing/For_Multi_Step_Prediction/y.csv', delimiter=',')
y = y.set_index("Time")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = False)

X_train = X_train[int(len(X_train)*1/2):]
X_test = X_test[:int(len(X_test)*1/2)]
y_train = y_train[int(len(y_train)*1/2):]
y_test = y_test[:int(len(y_test)*1/2)]
dates = dates[-len(X_test)-len(X_test)*2:-len(X_test)]

# Feature Scaling
x_scaler = StandardScaler()
y_scaler = StandardScaler()
X_train = x_scaler.fit_transform(X_train)
X_test = x_scaler.transform(X_test)
y_train = y_scaler.fit_transform(y_train)

########################################################################################################################
# Create the model.
########################################################################################################################

# Define the hyperparameters.
learning_rate = 0.001
number_of_epochs = 100
batch_size = 32

# Create the model.
my_model = create_model(X_train, learning_rate)

# Extract the loss per epoch to plot the learning progress.
hist_list = pd.DataFrame()

tscv = TimeSeriesSplit()
for train_index, test_index in tscv.split(X_train):
     X_train_split, X_test_split = X_train[train_index], X_train[test_index]
     y_train_split, y_test_split = y_train[train_index], y_train[test_index]
     X_train_split = np.reshape(X_train_split, (X_train_split.shape[0],X_train_split.shape[1],1))
     hist_split = train_model(my_model, X_train_split, y_train_split, number_of_epochs, batch_size)
     hist_list = hist_list.append(hist_split)

# Plot the loss per epoch.
metric = "mean_absolute_error"
plot_the_loss_curve(np.linspace(1,len(hist_list), len(hist_list) ), hist_list[metric], metric)

# my_model.save("Electricity_Generation_Prediction/LSTM/Single_Multi_Step_Prediction/SMST_No_Date.h5")
#my_model = keras.models.load_model("Electricity_Generation_Prediction/LSTM/Single_Multi_Step_Prediction/SMST_No_Date.h5")

########################################################################################################################
# Predicting the generation.
########################################################################################################################

pred_train = y_scaler.inverse_transform(my_model.predict(np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))))/1000
pred_train = pred_train.reshape(-1,)
pred_test = y_scaler.inverse_transform(my_model.predict(np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))))/1000
pred_test = pred_test.reshape(-1,)

X_train = x_scaler.inverse_transform(X_train)
X_train[:,0] = X_train[:,0]/1000
X_test = x_scaler.inverse_transform(X_test)
X_train[:,0] = X_train[:,0]/1000
y_train = (y_scaler.inverse_transform(y_train)/1000).reshape(-1,)
y_test = np.array(y_test.iloc[:,-1]/1000).reshape(-1,)

########################################################################################################################
# Data processing for plotting curves and printing the errors.
########################################################################################################################

# Compute the error between the Actual Generation and the prediction from the NN
print("-"*200)
error_train = pred_train - y_train
print("The mean absolute error of the train set is %0.2f" % mean_absolute_error(y_train,pred_train))
print("The mean squared error of the train set is %0.2f" % mean_squared_error(y_train,pred_train))
print("The root mean squared error of the train set is %0.2f" % np.sqrt(mean_squared_error(y_train,pred_train)))
print("-"*200)

error_test = pred_test - y_test
print("The mean absolute error of the test set is %0.2f" % mean_absolute_error(y_test,pred_test))
print("The mean squared error of the test set is %0.2f" % mean_squared_error(y_test,pred_test))
print("The root mean squared error of the test set is %0.2f" % np.sqrt(mean_squared_error(y_test,pred_test)))
print("-"*200)

########################################################################################################################
# Plotting curves.
########################################################################################################################

error_test_plot = np.zeros((48*3+48*7,1))
error_test_plot[-336:] = error_test[:48*7].reshape(-1,1)
# Plot the result with the truth in red and the predictions in blue.
fig2, axs2=plt.subplots(2,1,figsize=(12,6))
axs2[0].plot(dates.iloc[-len(X_test)-48*3:-len(X_test)],
             y_train[-48*3:],
             label = "Training Set (True Values)", alpha = 1, color = "blue")
axs2[0].plot(dates.iloc[-len(X_test):-len(X_test)+48*7],
             pred_test[:48*7],
             label = "LSTM Prediction", color = "orange")
axs2[0].plot(dates.iloc[-len(X_test):-len(X_test)+48*7],
             y_test[:48*7],
             label = "Test Set (True Values)", alpha = 1, color = "black")
axs2[0].axvline(dates.iloc[-len(X_test)], linestyle="--", color = "black")
axs2[0].set_ylabel('Load [GW]',size = 14)

axs2[1].plot(dates.iloc[-len(X_test)-48*3:-len(X_test)+48*7],
             error_test_plot,
             label = "Error", alpha = 1, color = "red")
axs2[1].axvline(dates.iloc[-len(X_test)], linestyle="--", color = "black")
axs2[1].set_xlabel('Date',size = 14)
axs2[1].set_ylabel('Error [GW]',size = 14)

# Include additional details such as tick intervals, rotation, legend positioning and grid on.
axs2[1].legend(loc=(1.04,0.9))
axs2[0].legend(loc=(1.04,0.7))
fig2.autofmt_xdate(rotation=12)
loc = plticker.MultipleLocator(base=47)
axs2[0].xaxis.set_major_locator(loc)
axs2[1].xaxis.set_major_locator(loc)
axs2[0].grid(True)
axs2[1].grid(True)

fig2.show()

########################################################################################################################
# Compute the standard deviation of the errors from the training set.
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
axs4.set_ylabel("Error during training [GW]", size = 14)
axs4.set_xlabel("Settlement Period / Weekday", size = 14)
loc = plticker.MultipleLocator(base=47) # Puts ticks at regular intervals
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
             label = "LSTM Pred.", color = "orange")
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
loc = plticker.MultipleLocator(base=47) # this locator puts ticks at regular intervals
axs5[0].xaxis.set_major_locator(loc)
axs5[1].xaxis.set_major_locator(loc)
fig5.autofmt_xdate(rotation=15)
axs5[1].legend(loc=(1.04,0.9))
axs5[0].legend(loc=(1.04,0.6))
fig5.show()

# Prediction on training set.
fig6, axs6=plt.subplots(3,1,figsize=(12,10))
# First plot contains the prediction, the true values from the test and training set and the standard deviation.
axs6[0].plot(dates_train[154:48*7*3+154+1],
             y_train[154:48*7*3+154+1],linewidth = 0.5,
             label = "Training Set \n(True Values)", alpha = 1, color = "blue")
axs6[0].plot(dates_train[154:48*7*3+154+1],
             pred_train[154:48*7*3+154+1], linewidth = 0.5,
             label = "LSTM Pred.", color = "orange")
axs6[0].set_ylabel('Load [GW]',size = 11)
axs6[0].set_xlabel('Date',size = 14)

# Second plot contains the errors.
axs6[1].plot(dates_train[154:48*7*3+154+1],
             error_train.iloc[154:48*7*3+154+1,-1],linewidth = 0.5,
             label = "Error", alpha = 1, color = "red")
axs6[1].set_xlabel('Date',size = 14)
axs6[1].set_ylabel('Error [GW]',size = 11)
loc1 = plticker.MultipleLocator(base=48*7) # this locator puts ticks at regular intervals
axs6[0].xaxis.set_major_locator(loc1)
axs6[1].xaxis.set_major_locator(loc1)

# Third plot contains the errors projected on a single week
axs6[2].scatter(error_train["SP"],
             error_train["Error_Train"],linewidth = 0.01,
             alpha=0.05, label = "Projected Errors", color = "red")
axs6[2].set_ylabel("Error during training [GW]", size = 11)
axs6[2].set_xticks(np.arange(1,385, 48))
axs6[2].set_xticklabels(["","1 / Monday", "49 / Tuesday", "97 / Wednesday", "145 / Thursday", "193 / Friday","241 / Saturday", "289 / Sunday",""])
axs6[2].set_xlabel("Settlement Period / Weekday", size = 14)

# Include additional details such as tick intervals, rotation, legend positioning and grid on.
axs6[2].grid(True)
axs6[1].grid(True)
axs6[0].grid(True)
loc2 = plticker.MultipleLocator(base=48) # this locator puts ticks at regular intervals
axs6[2].xaxis.set_major_locator(loc2)
axs6[2].legend(loc=(1.04,0.9))
axs6[1].legend(loc=(1.04,0.9))
axs6[0].legend(loc=(1.04,0.8))

fig6.show()

########################################################################################################################
# Save the results in a csv file.
########################################################################################################################

import csv
with open('Compare_Models/Single_Multi_Step_results/LSTM.csv', 'w', newline='', ) as file:
    writer = csv.writer(file)
    writer.writerow(["Method","MSE","MAE","RMSE"])
    writer.writerow(["LSTM",
                     str(mean_squared_error(y_test,pred_test)),
                     str(mean_absolute_error(y_test,pred_test)),
                     str(np.sqrt(mean_squared_error(y_test,pred_test)))
                     ])

import csv
with open('Compare_Models\SMST_Probability_results\Probability_Based_on_Training\LSTM_error.csv', 'w', newline='', ) as file:
    writer = csv.writer(file)
    writer.writerow(["Method","MSE","MAE","RMSE"])
    writer.writerow(["LSTM",
                     str(mean_squared_error(y_test,pred_test)),
                     str(mean_absolute_error(y_test,pred_test)),
                     str(np.sqrt(mean_squared_error(y_test,pred_test)))
                     ])

training_stats.to_csv('Compare_Models\SMST_Probability_results\Probability_Based_on_Training\LSTM_mean_errors_stddevs.csv')


