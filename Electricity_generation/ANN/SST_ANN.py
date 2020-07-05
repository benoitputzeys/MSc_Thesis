import numpy as np
import matplotlib.pyplot as plt
from Electricity_generation.ANN.Functions_ANN import plot_the_loss_curve, train_model, create_model
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import keras
########################################################################################################################
# Get data and data preprocessing.
########################################################################################################################

from numpy import genfromtxt

# Get the X (containing the features) and y (containing the labels) values
X = genfromtxt('Data_Entsoe/Data_Preprocessing/X.csv', delimiter=',')
y = genfromtxt('Data_Entsoe/Data_Preprocessing/y.csv', delimiter=',')
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
my_model = create_model(len(X_train[1]), learning_rate)

# Extract the loss per epoch to plot the learning progress.

hist_list = pd.DataFrame()

tscv = TimeSeriesSplit()
for train_index, test_index in tscv.split(X_train):
      X_train_split, X_test_split = X_train[train_index], X_train[test_index]
      y_train_split, y_test_split = y_train[train_index], y_train[test_index]
      hist_split = train_model(my_model, X_train_split, y_train_split, number_of_epochs, batch_size)
      hist_list = hist_list.append(hist_split)

# Plot the loss per epoch.
metric = "mean_absolute_error"
plot_the_loss_curve(np.linspace(1,len(hist_list), len(hist_list) ), hist_list[metric],metric)
my_model.save("my_model.h5")

#my_model = keras.models.load_model("my_model.h5")

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
print("The mean absolute error of the test set is %0.2f" % mean_absolute_error(y_scaler.inverse_transform(y_test),result_test))
print("The mean squared error of the test set is %0.2f" % mean_squared_error(y_scaler.inverse_transform(y_test),result_test))
print("The root mean squared error of the test set is %0.2f" % np.sqrt(mean_squared_error(y_scaler.inverse_transform(y_test),result_test)))
print("-"*200)

########################################################################################################################
# Plotting curves.
########################################################################################################################

# Plot the actual recorded generation against the date.
from Electricity_generation.ANN.Functions_ANN import plot_actual_generation, plot_predicted_generation, plot_error, plot_prediction_zoomed_in, plot_total_generation

fig1 = plt
plot_total_generation(fig1, X,y,"Total generation (Train + Test Set")

# Plot the actual recorded generation against the date.
fig2, axes1 = plt.subplots(3)
fig2.suptitle('Test Set (ANN)', fontsize=16)
# Plot the actual generation in a new subplot of 3x1.
plot_actual_generation(axes1, X[len(X)-len(y_test):,:], y[-len(y_test):], "Actual Generation")
# Plot the the predicted (NN) generation.
plot_predicted_generation(axes1, X[len(X)-len(result_test):,:], result_test, "NN prediction test set")
# Plot the error between the predicted and the actual temperature.
plot_error(axes1, X[len(X)-len(error_test):,:], error_test, "NN error test set")
fig2.show()

# Plot the prediction over the last 3 days.
plot_prediction_zoomed_in(X[-48*3:], result_test[-48*3:], "Prediction last 3 days")
plt.show()
########################################################################################################################
# Save the results in a csv file.
########################################################################################################################

import csv
with open('/Compare_Models/SST_results/ANN_result.csv', 'w', newline='', ) as file:
    writer = csv.writer(file)
    writer.writerow(["Method","MSE","MAE","RMSE"])
    writer.writerow(["ANN",
                     str(mean_squared_error(y_scaler.inverse_transform(y_test),result_test)),
                     str(mean_absolute_error(y_scaler.inverse_transform(y_test),result_test)),
                     str(np.sqrt(mean_squared_error(y_scaler.inverse_transform(y_test),result_test)))
                     ])
