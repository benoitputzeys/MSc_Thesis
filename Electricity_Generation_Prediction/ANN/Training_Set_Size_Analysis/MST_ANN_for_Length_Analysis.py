import numpy as np
import matplotlib.pyplot as plt
from Electricity_Generation_Prediction.ANN.Functions_ANN import plot_the_loss_curve, train_model, create_model
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.ticker as plticker
import keras

########################################################################################################################
# Get data and data preprocessing.
########################################################################################################################

# Get the X (containing the features) and y (containing the labels) values
X = pd.read_csv('Data_Preprocessing/For_336_SP_Step_Prediction/X.csv', delimiter=',')
X = X.set_index("Time")
dates = X.iloc[:,-1]
X = X.iloc[:,:-6]

y = pd.read_csv('Data_Preprocessing/For_336_SP_Step_Prediction/y.csv', delimiter=',')
y = y.set_index("Time")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = False)

# Save the unscaled data for later for data representation.
X_test_unscaled = X_test
X_train_unscaled = X_train

X_train = X_train[int(len(X_train)*3/4):]
y_train = y_train[int(len(y_train)*2/5):]

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
number_of_epochs = 160
batch_size = 19

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
plot_the_loss_curve(np.linspace(1,len(hist_list), len(hist_list) ), hist_list[metric])

my_model.save("Electricity_Generation_Prediction/ANN/Training_Set_Size_Analysis/Models/DMST_ANN_35L_Training_Set.h5")
#my_model = keras.models.load_model("Electricity_Generation_Prediction/ANN/Training_Set_Size_Analysis/Models/DMST_ANN_1L_Training_Set.h5")

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
# Plotting curves. For Info only, not really necessary!!!
########################################################################################################################
#
## Print the prediction of the first week in the test set.
#fig4, axes4 = plt.subplots(2,1,figsize=(12,6))
#axes4[0].plot(dates[-len(X_test):-len(X_test)+48*7],result_test[:48*7]/1000, label = "ANN Prediction", color = "orange")
#axes4[0].plot(dates[-len(X_test):-len(X_test)+48*7],y_test[:48*7]/1000, label = "Training Set", color = "blue")
#axes4[0].plot(30,30, label = "Error", color = "red")
#axes4[0].set_ylabel("Electricity Load, GW")
#axes4[0].grid(True)
#axes4[0].legend()
#plt.setp(axes4[0].get_xticklabels(), visible=False)
#
#axes4[1].plot(dates[-len(X_test):-len(X_test)+48*7],abs(result_test[:48*7]-(y_test[:48*7]))/1000, label = "Error", color = "red")
#axes4[1].set_ylabel("Error in, GW")
#
## Include additional details such as tick intervals, rotation, legend positioning and grid on.
#axes4[0].grid(True), axes4[1].grid(True)
#loc = plticker.MultipleLocator(base=48) # Puts ticks at regular intervals
#axes4[0].xaxis.set_major_locator(loc), axes4[1].xaxis.set_major_locator(loc)
#fig4.autofmt_xdate(rotation=0)
#axes4[0].legend(loc=(1.02,0.65)),
#
#plt.xticks(np.arange(1,48*7+2, 48), ["14:00\n07/25","14:00\n07/26","14:00\n07/27",
#                                  "14:00\n07/28","14:00\n07/29","14:00\n07/30",
#                                  "14:00\n07/31","14:00\n08/01"])
#
#fig4.show()
########################################################################################################################
# Save the results in a csv file.
########################################################################################################################


import csv
with open('Electricity_Generation_Prediction\ANN\Training_Set_Size_Analysis\AF_35L.csv', 'w', newline='',) as file:
    writer = csv.writer(file)
    writer.writerow(["Method","MSE","MAE","RMSE"])
    writer.writerow(["3/5 L",
                     str(mean_squared_error(y_test,result_test)),
                     str(mean_absolute_error(y_test,result_test)),
                     str(np.sqrt(mean_squared_error(y_test,result_test)))
                     ])
