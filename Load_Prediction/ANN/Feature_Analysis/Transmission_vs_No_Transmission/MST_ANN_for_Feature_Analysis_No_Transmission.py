import numpy as np
import matplotlib.pyplot as plt
from Load_Prediction.ANN.Functions_ANN import plot_the_loss_curve, train_model, create_model
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

X = pd.DataFrame(X)
X = X.drop(["Transmission_Past"], axis = 1)
X = np.array(X)

y = pd.read_csv('Data_Preprocessing/For_336_SP_Step_Prediction/y.csv', delimiter=',')
y = y.set_index("Time")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = False)

# Save the unscaled data for later for data representation.
X_test_unscaled = X_test
X_train_unscaled = X_train

X_train = X_train[int(len(X_train)*1/2):]
X_test = X_test[:int(len(X_test)*1/2)]
y_train = y_train[int(len(y_train)*1/2):]
y_test = y_test[:int(len(y_test)*1/2)]
dates = dates[-len(X_test)-len(X_test)*2:-len(X_test)]

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

my_model.save("Load_Prediction/ANN/Feature_Analysis/DMST_ANN_model_No_Transmission.h5")
#my_model = keras.models.load_model("Load_Prediction/ANN/Reature_Analysis/DMST_ANN_model_No_Transmission.h5")

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
error_train = (result_train - y_train)
print("The mean absolute error of the train set is %0.2f" % mean_absolute_error(y_train,result_train))
print("The mean squared error of the train set is %0.2f" % mean_squared_error(y_train,result_train))
print("The root mean squared error of the train set is %0.2f" % np.sqrt(mean_squared_error(y_train,result_train)))
print("-"*200)

error_test = (result_test - y_test)
print("The mean absolute error of the test set is %0.2f" % mean_absolute_error(y_test,result_test))
print("The mean squared error of the test set is %0.2f" % mean_squared_error(y_test,result_test))
print("The root mean squared error of the test set is %0.2f" % np.sqrt(mean_squared_error(y_test,result_test)))
print("-"*200)
########################################################################################################################
# Plotting curves.
########################################################################################################################

# Print the prediction of the training set.
fig2, axes2 = plt.subplots(2,1,figsize=(12,6))
axes2[0].plot(dates[-len(X_test)-48*7:-len(X_test)],
              result_train[-48*7:]/1000,
              label = "ANN Prediction Training Set", color = "orange")
axes2[0].plot(dates[-len(X_test)-48*7:-len(X_test)],
              y_train[-48*7:]/1000,
              label = "Training Set ", color = "blue")
axes2[0].set_ylabel("Electricity Load, GW")
loc = plticker.MultipleLocator(base=47) # this locator puts ticks at regular intervals
axes2[0].xaxis.set_major_locator(loc)
axes2[0].grid(True)
axes2[0].legend()
plt.setp(axes2[0].get_xticklabels(), visible=False)

axes2[1].plot(dates[-len(X_test)-48*7:-len(X_test)],
              (result_train[-48*7:]-y_train[-48*7:])/1000,
              label = "Error", color = "red")
axes2[1].set_xlabel("Date", size = 14)
axes2[1].set_ylabel("Error, GW")
axes2[1].xaxis.set_major_locator(loc)
axes2[1].legend()
axes2[1].grid(True)
fig2.show()

# Print the prediction of the first week in the test set.
fig4, axes4 = plt.subplots(2,1,figsize=(12,6))
axes4[0].plot(dates[-len(X_test):-len(X_test)+48*7],
              result_test[:48*7]/1000,
              label = "ANN Prediction Test Set", color = "orange")
axes4[0].plot(dates[-len(X_test):-len(X_test)+48*7],
              y_test[:48*7]/1000,
              label = "Test Set", color = "black")
axes4[0].set_ylabel("Electricity Load, GW")
loc = plticker.MultipleLocator(base=47) # Puts ticks at regular intervals
axes4[0].xaxis.set_major_locator(loc)
axes4[0].grid(True)
axes4[0].legend()

axes4[1].plot((result_test[:48*7]-(y_test[:48*7]))/1000, label = "Error", color = "red")
axes4[1].set_xlabel("Settlement Periods Test Set")
axes4[1].set_ylabel("Error in, GW")
axes4[1].xaxis.set_major_locator(loc)
axes4[1].grid(True)
axes4[1].legend()
fig4.show()

########################################################################################################################
# Save the results in a csv file.
########################################################################################################################

import csv
with open('Load_Prediction/ANN/Feature_Analysis/F6_(No_Transmission).csv', 'w', newline='',) as file:
    writer = csv.writer(file)
    writer.writerow(["Method","MSE","MAE","RMSE"])
    writer.writerow(["F6_(No_Transmission)",
                     str(mean_squared_error(y_test,result_test)),
                     str(mean_absolute_error(y_test,result_test)),
                     str(np.sqrt(mean_squared_error(y_test,result_test)))
                     ])

