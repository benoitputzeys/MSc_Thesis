import numpy as np
import matplotlib.pyplot as plt
from Electricity_generation.LSTM.Functions_LSTM import plot_the_loss_curve, train_model, create_model
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import keras
from scipy.ndimage.interpolation import shift
import datetime
from pandas import DataFrame

########################################################################################################################
# Get data and data preprocessing.
########################################################################################################################

from numpy import genfromtxt

def create_dates(features_df, y_values):

    date_list = [datetime.datetime(year=int(round(features_df[i, -1])),
                                   month=int(round(features_df[i, -2])),
                                   day=int(round(features_df[i, -3])),
                                   hour=int((features_df[i, -4] - 1) / 2),
                                   minute=int(((features_df[i, -4] -1) % 2 ) * 30)) for i in range(len(features_df))]

    df_dates = DataFrame(date_list, columns=['Date'])
    df_dates = df_dates.set_index(['Date'])
    df_dates['Load'] = y_values

    return df_dates


# Get the X (containing the features) and y (containing the labels) values
X = genfromtxt('Data_Entsoe/Data_Preprocessing/For_Multi_Step_Prediction/X.csv', delimiter=',')
y = genfromtxt('Data_Entsoe/Data_Preprocessing/For_Multi_Step_Prediction/y.csv', delimiter=',')
y = np.reshape(y, (len(y), 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = False)
X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(X_train, y_train, test_size = 0.2, random_state = 0, shuffle = False)
# Save the unscaled data for later for data representation.
X_test_unscaled = X_test
X_train_unscaled_1 = X_train_1

# Feature Scaling
x_scaler = StandardScaler()
y_scaler = StandardScaler()
X_train_1 = x_scaler.fit_transform(X_train_1)
X_train_2 = x_scaler.transform(X_train_2)
X_test = x_scaler.transform(X_test)
y_train_1 = y_scaler.fit_transform(y_train_1)

########################################################################################################################
# Create the model.
########################################################################################################################

# Define the hyperparameters.
learning_rate = 0.001
number_of_epochs = 50
batch_size = 32

# Create the model.
my_model = create_model(X_train_1, learning_rate)

# Extract the loss per epoch to plot the learning progress.
hist_list = pd.DataFrame()

tscv = TimeSeriesSplit()
for train_index, test_index in tscv.split(X_train_1):
      X_train_split, X_test_split = X_train_1[train_index], X_train_1[test_index]
      y_train_split, y_test_split = y_train_1[train_index], y_train_1[test_index]
      X_train_split = np.reshape(X_train_split, (X_train_split.shape[0],X_train_split.shape[1],1))
      hist_split = train_model(my_model, X_train_split, y_train_split, number_of_epochs, batch_size)
      hist_list = hist_list.append(hist_split)

# Plot the loss per epoch.
metric = "mean_absolute_error"
plot_the_loss_curve(np.linspace(1,len(hist_list), len(hist_list) ), hist_list[metric], metric)
my_model.save("my_model_MST_2.h5")

# my_model = keras.models.load_model("my_model_MST_2.h5")

########################################################################################################################
# Predicting the generation.
########################################################################################################################

result_train_1 = y_scaler.inverse_transform(my_model.predict(np.reshape(X_train_1, (X_train_1.shape[0],X_train_1.shape[1],1))))
result_train_2 = y_scaler.inverse_transform(my_model.predict(np.reshape(X_train_2, (X_train_2.shape[0],X_train_2.shape[1],1))))
result_test = y_scaler.inverse_transform(my_model.predict(np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))))

X_train_1 = x_scaler.inverse_transform(X_train_1)
X_train_2 = x_scaler.inverse_transform(X_train_2)
X_test = x_scaler.inverse_transform(X_test)
y_train_1 = y_scaler.inverse_transform(y_train_1)

########################################################################################################################
# Data processing for plotting curves and printing the errors.
########################################################################################################################

# Compute the error between the Actual Generation and the prediction from the NN
print("-"*200)
error_train_1 = abs(result_train_1 - y_train_1)
print("The mean absolute error of the training set is %0.2f" % mean_absolute_error(y_train_1,result_train_1))
print("The mean squared error of the training set is %0.2f" % mean_squared_error(y_train_1,result_train_1))
print("The root mean squared error of the training set is %0.2f" % np.sqrt(mean_squared_error(y_train_1,result_train_1)))

print("-"*200)
error_train_2 = abs(result_train_2 - y_train_2)
print("The mean absolute error of the train set 2 is %0.2f" % mean_absolute_error(y_train_2,result_train_2))
print("The mean squared error of the train set 2 is %0.2f" % mean_squared_error(y_train_2,result_train_2))
print("The root mean squared error of the train set 2 is %0.2f" % np.sqrt(mean_squared_error(y_train_2,result_train_2)))
print("-"*200)

error_test = abs(result_test - y_test)
print("The mean absolute error of the test set is %0.2f" % mean_absolute_error(y_test,result_test))
print("The mean squared error of the test set is %0.2f" % mean_squared_error(y_test,result_test))
print("The root mean squared error of the test set is %0.2f" % np.sqrt(mean_squared_error(y_test,result_test)))
print("-"*200)

########################################################################################################################
# Plotting curves.
########################################################################################################################

# Plot the actual recorded generation against the date.
from Electricity_generation.ANN.Functions_ANN import plot_actual_generation, plot_predicted_generation, plot_error, plot_prediction_zoomed_in, plot_total_generation

#plot_total_generation(X_train_1, y_train_1, "Total generation (Train + Test Set")

# Plot the actual recorded generation against the date.
fig, axes = plt.subplots(3)

fig.suptitle('Train Set 1 (LSTM)', fontsize=16)
# Plot the actual generation in a new subplot of 3x1.
plot_actual_generation(axes, X_train_1, y_train_1, "Actual Generation")

# Plot the the predicted (NN) generation.
plot_predicted_generation(axes, X_train_1, result_train_1, "NN prediction train set 1")

# Plot the error between the predicted and the actual temperature.
plot_error(axes, X_train_1, error_train_1, "NN error train set 1")

# Print the prediction of the training set 1.
y_values_dates = create_dates(X_train_1[-48*7:],result_train_1[-48*7:])
fig, axes = plt.subplots(2)
axes[0].plot(y_values_dates, label = "Prediction")
y_values_dates = create_dates(X_train_1[-48*7:],y_train_1[-48*7:])
axes[0].plot(y_values_dates, label = "Actual")
axes[0].set_xlabel("Settlement Periods Training Set 1")
axes[0].set_ylabel("Electricity Load [MW]")
axes[0].legend()

y_values_dates = create_dates(X_train_1[-48*7:],abs(result_train_1[-48*7:]-y_train_1[-48*7:]))
axes[1].plot(y_values_dates, label = "Error")
axes[1].set_xlabel("Settlement Periods Training Set 1")
axes[1].set_ylabel("Electricity Load [MW]")
axes[1].legend()

# Print the prediction of the training set 2.
fig1, axes1 = plt.subplots(2)
y_values_dates = create_dates(X_train_2[-48*7-1:],result_train_2[-48*7-1:])
axes1[0].plot(y_values_dates, label = "Prediction")
y_values_dates = create_dates(X_train_2[-48*7-1:],y_train_2[-48*7-1:])
axes1[0].plot(y_values_dates, label = "Actual")
axes1[0].set_xlabel("Settlement Periods Training Set 2")
axes1[0].set_ylabel("Electricity Load [MW]")
axes1[0].legend()

y_values_dates = create_dates(X_train_2[-48*7-1:],abs(result_train_2[-48*7-1:]-(y_train_2[-48*7-1:])))
axes1[1].plot(y_values_dates, label = "Error")
axes1[1].set_xlabel("Settlement Periods Training Set 2")
axes1[1].set_ylabel("Electricity Load [MW]")
axes1[1].legend()

# Print the prediction of the test set.
fig2, axes2 = plt.subplots(2)
y_values_dates = create_dates(X_test[-48*7:],result_test[-48*7:])
axes2[0].plot(y_values_dates, label = "Prediction")
y_values_dates = create_dates(X_test[-48*7:],y_test[-48*7:])
axes2[0].plot(y_values_dates, label = "Actual")
axes2[0].set_xlabel("Settlement Periods Test Set")
axes2[0].set_ylabel("Electricity Load [MW]")
axes2[0].legend()

y_values_dates = create_dates(X_test[-48*7:],abs(result_test[-48*7:]-(y_test[-48*7:])))
axes2[1].plot(y_values_dates, label = "Error")
axes2[1].set_xlabel("Settlement Periods Test Set")
axes2[1].set_ylabel("Error in [MW]")
axes2[1].legend()

########################################################################################################################
# Save the results in a csv file.
########################################################################################################################

pd.DataFrame(result_train_2).to_csv("Electricity_generation/Hybrid_Model/Pred_train2_other_metrics/LSTM_prediction.csv")
pd.DataFrame(result_test).to_csv("Electricity_generation/Hybrid_Model/Pred_test_other_metrics/LSTM_prediction.csv")

import csv
with open('/Compare_Models/MST2_results/LSTM_result.csv', 'w', newline='',) as file:
    writer = csv.writer(file)
    writer.writerow(["Method","MSE","MAE","RMSE"])
    writer.writerow(["LSTM",
                     str(mean_squared_error(y_test,result_test)),
                     str(mean_absolute_error(y_test,result_test)),
                     str(np.sqrt(mean_squared_error(y_test,result_test)))
                     ])
