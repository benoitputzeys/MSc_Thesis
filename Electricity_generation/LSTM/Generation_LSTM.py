import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from Functions_LSTM import plot_loss, train_model, create_model

########################################################################################################################

# Get data and data preprocessing.

########################################################################################################################

# Import the timeseries data
df = pd.read_csv("/Users/benoitputzeys/Desktop/Master Thesis/Data/SPI_2020/SPI_202005.csv")

df_label = df["Total_Generation"]
df_features = pd.DataFrame()
df_features["Total_Generation"] = df["Total_Generation"].shift(+2)
df_features["Settlement_Period"] = df["Settlement_Period"]

# Create your input variable
X = df_features.values
y = df_label.values
y = np.reshape(y,(len(y),1))

# After having shifted the data, the nan values have to be replaces in order to have good predictions.
replace_nan = SimpleImputer(missing_values = np.nan, strategy='mean')
replace_nan.fit(X[:,0:1])
X[:, 0:1] = replace_nan.transform(X[:,0:1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = False)
X_test_unscaled = X_test
X_train_unscaled = X_train

########################################################################################################################

# Create the model.

########################################################################################################################

# Define the hyperparameters.
learning_rate = 0.001
epochs = 50
batch_size = 65

# Create the model.
my_model = create_model(X_train, learning_rate)

# Extract the loss per epoch to plot the learning progress.
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
epochs, loss = train_model(my_model, X_train, y_train, epochs, batch_size)

# Plot the loss per epoch.
plot_loss(epochs[5:], loss[5:])

########################################################################################################################

# Predicting the generation.

########################################################################################################################

predicted_NN_generation_train = my_model.predict(X_train)
predicted_NN_generation_test = my_model.predict(X_test)

########################################################################################################################

# Data processing for plotting curves and printing the errors.

########################################################################################################################


# Create the Previous Day

error_previousday_train = abs(y[:len(X_train),0] - X_train[:,0,0])
print("The mean absolute error from the previous day prediction on the training set is %.2f" %np.average(error_previousday_train))
print("The mean squarred error from the previous day prediction on the training set is %.2f" %np.average(error_previousday_train*error_previousday_train))

error_previousday_test = abs(y[-len(X_test):,0] - X_test[:,0,0])
print("The mean absolute error from the previous day prediction on the testing set is %.2f" %np.average(error_previousday_test))
print("The mean squarred error from the previous day prediction on the testing set is %.2f" %np.average(error_previousday_test*error_previousday_test))

# Compute the error between the Actual Generation and the prediction from the NN
error_NN_train = abs(predicted_NN_generation_train[:,0] - y[:len(X_train),0])
print("The mean absolute error from the NN prediction on the training set is %.2f"  %np.mean(error_NN_train))
print("The mean squarred error from the NN prediction on the training set is %.2f"  %np.mean(error_NN_train*error_NN_train))

# Compute the error between the Actual Generation and the prediction from the NN
error_NN_test = abs(predicted_NN_generation_test[:,0] - y[-len(X_test):,0])
print("The mean absolute error from the NN prediction on the test set is %.2f"  %np.mean(error_NN_test))
print("The mean squarred error from the NN prediction on the test set is %.2f"  %np.mean(error_NN_test*error_NN_test))

########################################################################################################################

# Plotting curves.

########################################################################################################################

# Plot the actual recorded generation against the date.
from Functions_LSTM import plot_generation, plot_generation, plot_prediction_zoomed_in

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
plot_generation(axes2[0], y, "Actual Genration")

# Plot the the predicted (NN) generation.
plot_generation(axes2[1], predicted_NN_generation_test, "NN prediciton test set")

# Plot the error between the predicted and the actual temperature.
plot_generation(axes2[2], error_NN_test, "NN error test")

# Plot the predicted generation on the last 60 days.
plot_prediction_zoomed_in(predicted_NN_generation_test[-60:], y[-60:], X_test[-60:,0], "Predicted", "Actual", "Previous day")

