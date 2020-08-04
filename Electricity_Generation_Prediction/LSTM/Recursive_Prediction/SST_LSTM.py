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
X = pd.read_csv('Data_Preprocessing/For_1_SP_Step_Prediction/X.csv', delimiter=',')
X = X.set_index("Time")
X = X.drop(columns = "Transmission_Past")
dates = X.iloc[:,-1]
X = X.iloc[:,:-5]

y = pd.read_csv('Data_Preprocessing/For_1_SP_Step_Prediction/y.csv', delimiter=',')
y = y.set_index("Time")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = False)

X_train = X_train[int(len(X_train)*1/2):]
X_test = X_test[:int(len(X_test)*1/2)]
y_train = y_train[int(len(y_train)*1/2):]
y_test = y_test[:int(len(y_test)*1/2)]
dates = dates[-len(X_train)-len(X_test)*2:-len(X_test)]

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

my_model.save("Electricity_Generation_Prediction/LSTM/Recursive_Prediction/SST_No_Trans_No_Date.h5")

# Plot the loss per epoch.
metric = "mean_absolute_error"
plot_the_loss_curve(np.linspace(1,len(hist_list), len(hist_list) ), hist_list[metric], metric)

#my_model = keras.models.load_model("SST_No_Trans_No_Date.h5")

########################################################################################################################
# Predicting the generation.
########################################################################################################################

pred_train = y_scaler.inverse_transform(my_model.predict(np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))))/1000
pred_test = y_scaler.inverse_transform(my_model.predict(np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))))/1000

X_train = x_scaler.inverse_transform(X_train)/1000
X_test = x_scaler.inverse_transform(X_test)/1000
y_train = y_scaler.inverse_transform(y_train)/1000
y_test = y_test/1000

########################################################################################################################
# Data processing for plotting curves and printing the errors.
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
# Plotting curves.
########################################################################################################################

error_test_plot = np.zeros((48*3+48*7,1))
error_test_plot[-336:] = error_test[:48*7]
# Plot the result with the truth in red and the predictions in blue.
fig2, axs2=plt.subplots(2,1,figsize=(12,6))
axs2[0].grid(True)
axs2[0].plot(dates.iloc[-len(X_test)-48*3:-len(X_test)],
             y_train[-48*3:,0],
             label = "Training Set", alpha = 1, color = "blue")
axs2[0].plot(dates.iloc[-len(X_test):-len(X_test)+48*7],
             pred_test[:48*7,0], label = "Single Step \nLSTM Prediction", color = "orange")
axs2[0].plot(dates.iloc[-len(X_test):-len(X_test)+48*7],
             y_test[:48*7],
             label = "Test Set", alpha = 1, color = "black")
axs2[0].axvline(dates.iloc[-len(X_test)], linestyle="--", color = "black")
axs2[0].set_ylabel('Load [GW]',size = 14)
loc = plticker.MultipleLocator(base=47) # this locator puts ticks at regular intervals

axs2[1].grid(True)
axs2[1].plot(dates.iloc[-len(X_test)-48*3:-len(X_test)+48*7],
             error_test_plot,
             label = "Error", alpha = 1, color = "red")
axs2[1].axvline(dates.iloc[-len(X_test)], linestyle="--", color = "black")
axs2[1].set_xlabel('Date',size = 14)
axs2[1].set_ylabel('Error [GW]',size = 14)

loc = plticker.MultipleLocator(base=47) # this locator puts ticks at regular intervals
axs2[1].xaxis.set_major_locator(loc)
axs2[0].xaxis.set_major_locator(loc)
fig2.autofmt_xdate(rotation=10)
axs2[1].legend(loc=(1.04,0.9))
axs2[0].legend(loc=(1.04,0.7))

fig2.show()

# ########################################################################################################################
# # Save the results in a csv file.
# ########################################################################################################################
#
# import csv
# with open('Compare_Models/Single_Step_Results/LSTM_result.csv', 'w', newline='', ) as file:
#     writer = csv.writer(file)
#     writer.writerow(["Method","MSE","MAE","RMSE"])
#     writer.writerow(["LSTM",
#                      str(mean_squared_error(y_scaler.inverse_transform(y_test),result_test)),
#                      str(mean_absolute_error(y_scaler.inverse_transform(y_test),result_test)),
#                      str(np.sqrt(mean_squared_error(y_scaler.inverse_transform(y_test),result_test)))
#                      ])

