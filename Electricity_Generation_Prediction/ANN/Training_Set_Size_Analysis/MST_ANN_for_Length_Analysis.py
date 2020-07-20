import numpy as np
import matplotlib.pyplot as plt
from Electricity_Generation_Prediction.ANN.Functions_ANN import plot_the_loss_curve, train_model, create_model
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

# Get the X (containing the features) and y (containing the labels) values
X = pd.read_csv('Data_Preprocessing/For_Multi_Step_Prediction/X.csv', delimiter=',')
X = X.set_index("Time")
X = X.iloc[:,:-1]

y = pd.read_csv('Data_Preprocessing/For_Multi_Step_Prediction/y.csv', delimiter=',')
y = y.set_index("Time")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = False)

# Save the unscaled data for later for data representation.
X_test_unscaled = X_test
X_train_unscaled = X_train

X_train = X_train[int(len(X_train)*1/2):]
y_train = y_train[int(len(y_train)*3/4):]

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

plot_the_loss_curve(np.linspace(1,len(hist_list), len(hist_list) ), hist_list[metric], metric)
my_model.save("my_model_MST_2.h5")

#my_model = keras.models.load_model("Electricity_Generation_Prediction/ANN/my_model_MST_2.h5")

########################################################################################################################
# Predicting the generation.
########################################################################################################################

result_train = y_scaler.inverse_transform(my_model.predict(X_train))
result_test = y_scaler.inverse_transform(my_model.predict(X_test))

X_train = x_scaler.inverse_transform(X_train)
X_test = x_scaler.inverse_transform(X_test)
y_train = y_scaler.inverse_transform(y_train)

########################################################################################################################
# Data processing for plotting curves and printing the errors.
########################################################################################################################

# Compute the error between the Actual Generation and the prediction from the NN

print("-"*200)
error_train = abs(result_train - y_train)
print("The mean absolute error of the train set is %0.2f" % mean_absolute_error(y_train,result_train))
print("The mean squared error of the train set is %0.2f" % mean_squared_error(y_train,result_train))
print("The root mean squared error of the train set is %0.2f" % np.sqrt(mean_squared_error(y_train,result_train)))
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
from Electricity_Generation_Prediction.ANN.Functions_ANN import plot_actual_generation, plot_predicted_generation, plot_error, plot_prediction_zoomed_in, plot_total_generation, create_dates

#plot_total_generation(X_train_1, y_train_1, "Total generation (Train + Test Set")

# Plot the actual recorded generation against the date.
fig1, axes1 = plt.subplots(3,1,figsize=(12,6))
#fig1.suptitle('Train Set (ANN)', fontsize=16)
# Plot the actual generation in a new subplot of 3x1.
plot_actual_generation(axes1, X_train, y_train, "Actual Generation")
# Plot the the predicted (NN) generation.
plot_predicted_generation(axes1, X_train, result_train, "NN prediction train set")
# Plot the error between the predicted and the actual temperature.
plot_error(axes1, X_train, error_train, "NN error train set")
fig1.show()

# Print the prediction of the training set.
y_values_dates = create_dates(X_train[-48*7:],result_train[-48*7:])
fig2, axes2 = plt.subplots(2,1,figsize=(12,6))
axes2[0].plot( y_values_dates/1000, label = "Prediction Train", color = "orange")
y_values_dates = create_dates(X_train[-48*7:],y_train[-48*7:])
axes2[0].plot( y_values_dates/1000, label = "Actual", color = "blue")
axes2[0].set_ylabel("Electricity Load [GW]")
axes2[0].grid(True)
axes2[0].legend()
plt.setp(axes2[0].get_xticklabels(), visible=False)

y_values_dates = create_dates(X_train[-48*7:],result_train[-48*7:]-y_train[-48*7:])
axes2[1].plot(y_values_dates/1000, label = "Error", color = "red")
axes2[1].set_xlabel("Date", size = 14)
axes2[1].set_ylabel("Error [GW]")
axes2[1].legend()
axes2[1].grid(True)
fig2.show()

# Print the prediction of the first week in the test set.
fig4, axes4 = plt.subplots(2,1,figsize=(12,6))
y_values_dates = create_dates(X_test[:48*7,:],result_test[:48*7])
axes4[0].plot(y_values_dates/1000, label = "Prediction Test", color = "orange")
y_values_dates = create_dates(X_test[:48*7,:],y_test[:48*7])
axes4[0].plot(y_values_dates/1000, label = "Actual", color = "blue")
# for tick in axes4[0].get_xticklabels():
#     tick.set_rotation(20)
axes4[0].set_ylabel("Electricity Load [GW]")
axes4[0].grid(True)
axes4[0].legend()
plt.setp(axes4[0].get_xticklabels(), visible=False)

y_values_dates = create_dates(X_test[:48*7,:],abs(result_test[:48*7]-(y_test[:48*7])))
axes4[1].plot(y_values_dates/1000, label = "Error", color = "red")
axes4[1].set_xlabel("Settlement Periods Test Set")
# for tick in axes4[1].get_xticklabels():
#     tick.set_rotation(20)
axes4[1].set_ylabel("Error in [GW]")
axes4[1].grid(True)
axes4[1].legend()
fig4.show()

########################################################################################################################
# Save the results in a csv file.
########################################################################################################################


import csv
with open('Electricity_Generation_Prediction\ANN\Training_Set_Size_Analysis\AF_14L.csv', 'w', newline='',) as file:
    writer = csv.writer(file)
    writer.writerow(["Method","MSE","MAE","RMSE"])
    writer.writerow(["1/4 L",
                     str(mean_squared_error(y_test,result_test)),
                     str(mean_absolute_error(y_test,result_test)),
                     str(np.sqrt(mean_squared_error(y_test,result_test)))
                     ])

#my_model.save("my_model.h5")
