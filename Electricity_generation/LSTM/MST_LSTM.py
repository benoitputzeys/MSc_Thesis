import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from Electricity_generation.LSTM.Functions_LSTM import plot_the_loss_curve, train_model, create_model, plot_generation, plot_prediction_zoomed_in
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pandas import DataFrame

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

# Define the hyperparameters.
learning_rate = 0.001
number_of_epochs = 50
batch_size = 124

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
plot_the_loss_curve(np.linspace(1,len(hist_list), len(hist_list) ), hist_list['mean_absolute_error'])

########################################################################################################################
# Predicting the generation.
########################################################################################################################

result_train = y_scaler.inverse_transform(my_model.predict(X_train))
result_test = y_scaler.inverse_transform(my_model.predict(X_test))

# Multi-Step
X_future_features = pd.DataFrame(data=X_test_unscaled,  columns=["0","1","2","3","4","5"])
DoW_SP = genfromtxt(
    '/Users/benoitputzeys/PycharmProjects/MSc_Thesis/Data_Entsoe/Data_Preprocessing/For_Multi_Step_Prediction/DoW_SP_2.csv',
    delimiter=',')
result_future = y_scaler.inverse_transform(y_test)

for i in range(0,48*7):

    prev_value = result_future[-2]

    new_row = [[prev_value[0], 0, 0, 0, 0, 0]]
    new_row = DataFrame(new_row, columns=["0", "1", "2", "3", "4", "5"])
    X_future_features = pd.concat([X_future_features,new_row], axis=0)

    rolling_mean_10 = X_future_features["0"].rolling(window=10).mean().values[-1]
    rolling_mean_50 = X_future_features["0"].rolling(window=50).mean().values[-1]
    exp_20 = X_future_features["0"].ewm(span=20, adjust=False).mean().values[-1]
    exp_50 = X_future_features["0"].ewm(span=50, adjust=False).mean().values[-1]

    update_row = [[prev_value, rolling_mean_10, rolling_mean_50, exp_20, exp_50, DoW_SP[i]]]

    update_row = DataFrame(update_row, columns=["0", "1", "2", "3", "4", "5"])
    X_future_features.iloc[-1,:] = update_row.iloc[0,:]

    result_future = np.append(result_future, y_scaler.inverse_transform(my_model.predict(X_scaler.transform(update_row).reshape(1,6,1))))
    result_future = np.reshape(result_future,(-1,1))

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

fig.suptitle('Training + Test Set (ANN)', fontsize=16)
# Plot the actual generation in a new subplot of 3x1.
plot_generation(axes[0], y[-len(result_test):], "Actual Generation")

# Plot the the predicted (NN) generation.
plot_generation(axes[1], result_test, "NN prediction test set")

# Plot the error between the predicted and the actual temperature.
plot_generation(axes[2], error_test, "NN error test")

# Plot the predicted generation on the last 60 days.
plot_prediction_zoomed_in(result_test[-60:], y[-60:], X_test_unscaled[-60:,0], "Predicted", "Actual", "Previous day")

fig3, ax3 = plt.subplots(2)
fig3.suptitle('LSTM: Future Evaluation', fontsize=16)
ax3[0].plot(result_future[-48*7:], linewidth=0.5)
ax3[0].set_xlabel('Settlement Period')
ax3[0].set_ylabel('Prediction')

ax3[1].plot(result_test[-48*7:],linewidth=0.5)
ax3[1].set_xlabel('Settlement Period')
ax3[1].set_ylabel('Prediction on test set')

########################################################################################################################
# Save the results in a csv file.
########################################################################################################################

import csv
with open('/Users/benoitputzeys/PycharmProjects/MSc_Thesis/Compare_Models/LSTM_result.csv', 'w', newline='',) as file:
    writer = csv.writer(file)
    writer.writerow(["Method","MSE","MAE","RMSE"])
    writer.writerow(["LSTM",
                     str(mean_squared_error(y_scaler.inverse_transform(y_test),result_test)),
                     str(mean_absolute_error(y_scaler.inverse_transform(y_test),result_test)),
                     str(np.sqrt(mean_squared_error(y_scaler.inverse_transform(y_test),result_test)))
                     ])

df_best = pd.read_csv("/Users/benoitputzeys/PycharmProjects/MSc_Thesis/Compare_Models/Best_Results/LSTM_result.csv")

import shutil
if mean_squared_error(y_scaler.inverse_transform(y_test), result_test) <= df_best.iloc[0,1]:
    import csv
    with open('/Users/benoitputzeys/PycharmProjects/MSc_Thesis/Compare_Models/Best_Results/LSTM_result.csv', 'w',newline='', ) as file:
        writer = csv.writer(file)
        writer.writerow(["Method", "MSE", "MAE","RMSE"])
        writer.writerow(["LSTM",
                         str(mean_squared_error(y_scaler.inverse_transform(y_test), result_test)),
                         str(mean_absolute_error(y_scaler.inverse_transform(y_test), result_test)),
                         str(np.sqrt(mean_squared_error(y_scaler.inverse_transform(y_test), result_test)))
                         ])
    shutil.copyfile('Generation_LSTM.py', 'Best_LSTM.py')