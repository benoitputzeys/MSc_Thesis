import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from Electricity_generation.LSTM.Functions_LSTM import plot_the_loss_curve, train_model, create_model, plot_generation, plot_prediction_zoomed_in
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import keras
########################################################################################################################
# Get data and data preprocessing.
########################################################################################################################

from numpy import genfromtxt

# Get the X (containing the features) and y (containing the labels) values
X = genfromtxt('/Users/benoitputzeys/PycharmProjects/MSc_Thesis/Data_Entsoe/Data_Preprocessing/X.csv', delimiter=',')
y = genfromtxt('/Users/benoitputzeys/PycharmProjects/MSc_Thesis/Data_Entsoe/Data_Preprocessing/y.csv', delimiter=',')
y = np.reshape(y, (len(y), 1))

# Split data into train set and test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle = False)
X_test_unscaled = X_test
X_train_unscaled = X_train

# Feature Scaling
X_scaler = StandardScaler()
y_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train)
X_test = X_scaler.transform(X_test)
y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.transform(y_test)
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

########################################################################################################################
# Create the model.
########################################################################################################################
#
# # Define the hyperparameters.
# learning_rate = 0.001
# number_of_epochs = 50
# batch_size = 64
#
# # Create the model.
# my_model = create_model(X_train, learning_rate)
#
# # Extract the loss per epoch to plot the learning progress.
# hist_list = pd.DataFrame()
#
# tscv = TimeSeriesSplit()
# for train_index, test_index in tscv.split(X_train):
#       X_train_split, X_test_split = X_train[train_index], X_train[test_index]
#       y_train_split, y_test_split = y_train[train_index], y_train[test_index]
#       X_train_split = np.reshape(X_train_split, (X_train_split.shape[0],X_train_split.shape[1],1))
#       hist_split = train_model(my_model, X_train_split, y_train_split, number_of_epochs, batch_size)
#       hist_list = hist_list.append(hist_split)
#
# # Plot the loss per epoch.
# metric = "mean_absolute_error"
# plot_the_loss_curve(np.linspace(1,len(hist_list), len(hist_list) ), hist_list[metric],metric)
my_model = keras.models.load_model("my_model.h5")

########################################################################################################################
# Predicting the generation.
########################################################################################################################

result_train = y_scaler.inverse_transform(my_model.predict(X_train))
result_test = y_scaler.inverse_transform(my_model.predict(X_test))

########################################################################################################################
# Data processing for plotting curves and printing the errors.
########################################################################################################################

# Compute the error between the Actual Generation and the prediction from the NN

print("-"*200)
error_train = abs(result_train[:,0] - y[:len(X_train),0])
print("The mean absolute error of the training set is %0.2f" % mean_absolute_error(y_scaler.inverse_transform(y_train),result_train))
print("The mean squared error of the training set is %0.2f" % mean_squared_error(y_scaler.inverse_transform(y_train),result_train))
print("The root mean squared error of the training set is %0.2f" % np.sqrt(mean_squared_error(y_scaler.inverse_transform(y_train),result_train)))

print("-"*200)
error_test = abs(result_test[:,0] - y[-len(X_test):,0])
print("The mean absolute error of the training set is %0.2f" % mean_absolute_error(y_scaler.inverse_transform(y_test),result_test))
print("The mean squared error of the training set is %0.2f" % mean_squared_error(y_scaler.inverse_transform(y_test),result_test))
print("The root mean squared error of the training set is %0.2f" % np.sqrt(mean_squared_error(y_scaler.inverse_transform(y_test),result_test)))

########################################################################################################################
# Plotting curves.
########################################################################################################################

# Plot the actual recorded generation against the date.
fig, axes = plt.subplots(3)

fig.suptitle('Test Set (LSTM)', fontsize=16)
# Plot the actual generation in a new subplot of 3x1.
plot_generation(axes[0], X_test_unscaled, y[-len(result_test):], "Actual Generation")

# Plot the the predicted (NN) generation.
plot_generation(axes[1], X_test_unscaled, result_test, "NN prediction test set")

# Plot the error between the predicted and the actual temperature.
plot_generation(axes[2], X_test_unscaled, error_test, "NN error test")

# Plot the predicted generation on the last 3 days.
fig1, axes1 = plt.subplots(2)
plot_prediction_zoomed_in(fig1, axes1, X_test_unscaled[-3*48:],result_test[-3*48:], y[-3*48:])

########################################################################################################################
# Save the results in a csv file.
########################################################################################################################

import csv
with open('/Compare_Models/SST_results/LSTM_result.csv', 'w', newline='', ) as file:
    writer = csv.writer(file)
    writer.writerow(["Method","MSE","MAE","RMSE"])
    writer.writerow(["LSTM",
                     str(mean_squared_error(y_scaler.inverse_transform(y_test),result_test)),
                     str(mean_absolute_error(y_scaler.inverse_transform(y_test),result_test)),
                     str(np.sqrt(mean_squared_error(y_scaler.inverse_transform(y_test),result_test)))
                     ])

df_best = pd.read_csv("/Users/benoitputzeys/PycharmProjects/MSc_Thesis/Compare_Models/Best_Results/LSTM_result.csv")
