import numpy as np
import matplotlib.pyplot as plt
from Electricity_generation.LSTM.Functions_LSTM import plot_the_loss_curve, train_model, create_model
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import keras
from scipy.ndimage.interpolation import shift

########################################################################################################################
# Get data and data preprocessing.
########################################################################################################################

from numpy import genfromtxt

# Get the X (containing the features) and y (containing the labels) values
X = genfromtxt('/Users/benoitputzeys/PycharmProjects/MSc_Thesis/Data_Entsoe/Data_Preprocessing/For_Multi_Step_Prediction_Outside_Test_Set/X.csv', delimiter=',')
y = genfromtxt('/Users/benoitputzeys/PycharmProjects/MSc_Thesis/Data_Entsoe/Data_Preprocessing/For_Multi_Step_Prediction_Outside_Test_Set/y.csv', delimiter=',')
y = np.reshape(y, (len(y), 1))


# Split data into train set and test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle=False)

# Save the unscaled data for later for data representation.
X_test_unscaled = X_test
X_train_unscaled = X_train

# Feature Scaling
x_scaler = StandardScaler()
y_scaler = StandardScaler()
X_train = x_scaler.fit_transform(X_train)
X_test = x_scaler.transform(X_test)
y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.transform(y_test)

########################################################################################################################
# Create the model.
########################################################################################################################

# Define the hyperparameters.
learning_rate = 0.001
number_of_epochs = 50
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

my_model.save("my_model_MST_2.h5")

########################################################################################################################
# Predicting the generation.
########################################################################################################################
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

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
print("The mean absolute error of the test set is %0.2f" % mean_absolute_error(y_scaler.inverse_transform(y_test),result_test))
print("The mean squared error of the test set is %0.2f" % mean_squared_error(y_scaler.inverse_transform(y_test),result_test))
print("The root mean squared error of the test set is %0.2f" % np.sqrt(mean_squared_error(y_scaler.inverse_transform(y_test),result_test)))
print("-"*200)

########################################################################################################################
# Plotting curves.
########################################################################################################################

# Plot the actual recorded generation against the date.
from Electricity_generation.ANN.Functions_ANN import plot_actual_generation, plot_predicted_generation, plot_error, plot_prediction_zoomed_in, plot_total_generation

plot_total_generation(X,y,"Total generation (Train + Test Set")

# Plot the actual recorded generation against the date.
fig, axes = plt.subplots(3)

fig.suptitle('Test Set (LSTM)', fontsize=16)
# Plot the actual generation in a new subplot of 3x1.
plot_actual_generation(axes, X[len(X)-len(y_test):,:], y[-len(y_test):], "Actual Generation")

# Plot the the predicted (NN) generation.
plot_predicted_generation(axes, X[len(X)-len(result_test):,:], result_test, "NN prediction test set")

# Plot the error between the predicted and the actual temperature.
plot_error(axes, X[len(X)-len(error_test):,:], error_test, "NN error test set")

# Plot the prediction over the last 3 days.
plot_prediction_zoomed_in(X[-48*3:], result_test[-48*3:], "Prediction last 3 days")

fig, axes = plt.subplots(2)
axes[0].plot(result_train[-48*7:], label = "Prediction")
axes[0].plot(y_scaler.inverse_transform(y_train[-48*7:]), label = "Actual")
axes[0].set_xlabel("Settlement Periods")
axes[0].set_ylabel("Electricity Load [MW]")
axes[0].legend()

axes[1].plot(abs(result_train[-48*7:]-y_scaler.inverse_transform(y_train[-48*7:])), label = "Error")
axes[1].set_xlabel("Settlement Periods")
axes[1].set_ylabel("Electricity Load [MW]")
axes[1].legend()

fig1, axes1 = plt.subplots(2)
axes1[0].plot(result_test[-48*7:], label = "Prediction")
axes1[0].plot(y_scaler.inverse_transform(y_test[-48*7:]), label = "Actual")
axes1[0].set_xlabel("Settlement Periods")
axes1[0].set_ylabel("Electricity Load [MW]")
axes1[0].legend()

axes1[1].plot(abs(result_test[-48*7:]-y_scaler.inverse_transform(y_test[-48*7:])), label = "Error")
axes1[1].set_xlabel("Settlement Periods")
axes1[1].set_ylabel("Electricity Load [MW]")
axes1[1].legend()

########################################################################################################################
# Save the results in a csv file.
########################################################################################################################

# import csv
# with open('/Users/benoitputzeys/PycharmProjects/MSc_Thesis/Compare_Models/ANN_result.csv', 'w', newline='',) as file:
#     writer = csv.writer(file)
#     writer.writerow(["Method","MSE","MAE","RMSE"])
#     writer.writerow(["ANN",
#                      str(mean_squared_error(y_scaler.inverse_transform(y_test),result_test)),
#                      str(mean_absolute_error(y_scaler.inverse_transform(y_test),result_test)),
#                      str(np.sqrt(mean_squared_error(y_scaler.inverse_transform(y_test),result_test)))
#                      ])
#
# df_best = pd.read_csv("/Users/benoitputzeys/PycharmProjects/MSc_Thesis/Compare_Models/Best_Results/ANN_result.csv")

#my_model.save("my_model.h5")
