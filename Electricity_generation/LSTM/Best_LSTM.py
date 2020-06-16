import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from Electricity_generation.LSTM.Functions_LSTM import plot_loss, train_model, create_model, plot_generation, plot_prediction_zoomed_in
from sklearn.preprocessing import StandardScaler
import pandas as pd

########################################################################################################################

# Get data and data preprocessing.

########################################################################################################################

from Data_Preprocessing.get_features_and_label import return_features_and_labels

# Get the X (containing the features) and y (containing the labels) values
X, y = return_features_and_labels()

# Split data into train set and test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = False)
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

# Define the hyperparameters.
learning_rate = 0.001
number_of_epochs = 50
batch_size = 32

# Create the model.
my_model = create_model(X_train, learning_rate)

# Extract the loss per epoch to plot the learning progress.

loss_list = np.empty
epochs_list = np.empty

tscv = TimeSeriesSplit()
for train_index, test_index in tscv.split(X_train):
      X_train_split, X_test_split = X_train[train_index], X_train[test_index]
      y_train_split, y_test_split = y_train[train_index], y_train[test_index]
      X_train_split = np.reshape(X_train_split, (X_train_split.shape[0],X_train_split.shape[1],1))
      epochs_split, loss = train_model(my_model, X_train_split, y_train_split, number_of_epochs, batch_size)
      epochs_list = np.append(epochs_list , epochs_split)
      loss_list = np.append(loss_list, loss)

epochs_list = np.delete(epochs_list, [0])
loss_list = np.delete(loss_list, [0])

# Plot the loss per epoch.
plot_loss(np.linspace(1,len(epochs_list), len(epochs_list) ), loss_list)

########################################################################################################################

# Predicting the generation.

########################################################################################################################

predicted_NN_generation_train = y_scaler.inverse_transform(my_model.predict(X_train))
predicted_NN_generation_test = y_scaler.inverse_transform(my_model.predict(X_test))

########################################################################################################################

# Data processing for plotting curves and printing the errors.

########################################################################################################################

# Create the Previous Day

error_previousday_train = abs(y[:len(X_train),0] - X_train_unscaled[:,0])
print("The mean absolute error from the previous day prediction on the training set is %.2f" %np.average(error_previousday_train))
print("The mean squarred error from the previous day prediction on the training set is %.2f" %np.average(error_previousday_train*error_previousday_train))

error_previousday_test = abs(y[-len(X_test):,0] - X_test_unscaled[:,0])
print("The mean absolute error from the previous day prediction on the testing set is %.2f" %np.average(error_previousday_test))
print("The mean squarred error from the previous day prediction on the testing set is %.2f" %np.average(error_previousday_test*error_previousday_test))

# Compute the error between the Actual Generation and the prediction from the NN
error_NN_train = abs(predicted_NN_generation_train[:,0] - y[:len(X_train_unscaled),0])
print("The mean absolute error from the NN prediction on the training set is %.2f"  %np.mean(error_NN_train))
print("The mean squarred error from the NN prediction on the training set is %.2f"  %np.mean(error_NN_train*error_NN_train))

# Compute the error between the Actual Generation and the prediction from the NN
error_NN_test = abs(predicted_NN_generation_test[:,0] - y[-len(X_test_unscaled):,0])
print("The mean absolute error from the NN prediction on the test set is %.2f"  %np.mean(error_NN_test))
print("The mean squarred error from the NN prediction on the test set is %.2f"  %np.mean(error_NN_test*error_NN_test))

########################################################################################################################

# Plotting curves.

########################################################################################################################

fig, axes = plt.subplots(3)

# Plot the actual generation in a subplot of 3x1.
plot_generation(axes[0], y, "Actual Generation")

# Plot the predicted generation (from the day before) against the recording date.
plot_generation(axes[1], X[:, 0], "Previous day prediction")

# Plot the error between the previous day and the actual generation.
plot_generation(axes[2], error_previousday_test, "Error previous day on test set.")

# Plot the actual recorded generation against the date.
fig2, axes2 = plt.subplots(3)

# Plot the actual generation in a new subplot of 3x1.
plot_generation(axes2[0], y[-len(predicted_NN_generation_test):], "Actual Genration")

# Plot the the predicted (NN) generation.
plot_generation(axes2[1], predicted_NN_generation_test, "NN prediciton test set")

# Plot the error between the predicted and the actual temperature.
plot_generation(axes2[2], error_NN_test, "NN error test")

# Plot the predicted generation on the last 60 days.
plot_prediction_zoomed_in(predicted_NN_generation_test[-60:], y[-60:], X_test_unscaled[-60:,0], "Predicted", "Actual", "Previous day")

########################################################################################################################

# Save the results in a csv file.

########################################################################################################################

import csv
with open('/Users/benoitputzeys/PycharmProjects/NN-Predicitons/Compare_Models/LSTM_result.csv', 'w', newline='',) as file:
    writer = csv.writer(file)
    writer.writerow(["Method","MSE","MAE"])
    writer.writerow(["LSTM",str(np.mean(error_NN_test*error_NN_test)),str(np.mean(error_NN_test))])

df_best = pd.read_csv("/Users/benoitputzeys/PycharmProjects/NN-Predicitons/Compare_Models/Best_Results/ANN_result.csv")

import shutil
if np.mean(error_NN_test*error_NN_test) <= df_best.iloc[0,1]:
    import csv
    with open('/Users/benoitputzeys/PycharmProjects/NN-Predicitons/Compare_Models/Best_Results/LSTM_result.csv', 'w',newline='', ) as file:
        writer = csv.writer(file)
        writer.writerow(["Method", "MSE", "MAE"])
        writer.writerow(["LSTM", str(np.mean(error_NN_test * error_NN_test)), str(np.mean(error_NN_test))])
    shutil.copyfile('Generation_ANN.py', 'Best_ANN.py')