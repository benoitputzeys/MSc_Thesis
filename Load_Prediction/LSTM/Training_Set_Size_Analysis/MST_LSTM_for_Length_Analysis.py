import numpy as np
import matplotlib.pyplot as plt
from Load_Prediction.LSTM.Functions_LSTM import train_model, create_model
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
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

# Partition the data into 80% training set and 20% test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = False)

# Save the unscaled data for later for data representation.
X_test_unscaled = X_test
X_train_unscaled = X_train

# Analyse the length of the training set by setting the length here:
X_train = X_train[int(len(X_train)*1/4):]
y_train = y_train[int(len(y_train)*1/4):]

# Feature Scaling
x_scaler = StandardScaler()
y_scaler = StandardScaler()
X_train = x_scaler.fit_transform(X_train)
X_test = x_scaler.transform(X_test)
y_train = y_scaler.fit_transform(y_train)

########################################################################################################################
# Create the model.
########################################################################################################################

# This code was used to create the models in the folder "Models" of this directory. As the models are already created,
# this does not have to be done again.

## Define the hyperparameters.
#learning_rate = 0.005
#number_of_epochs = 40
#batch_size = 29
#
## Create the model.
#my_model = create_model(X_train, learning_rate)
#
## Create a dataframe to extract the loss per epoch to plot the learning progress.
#
#hist_list = pd.DataFrame()
#
## Perform 5-fold cross validation
#tscv = TimeSeriesSplit()
#for train_index, test_index in tscv.split(X_train):
#     X_train_split, X_test_split = X_train[train_index], X_train[test_index]
#     y_train_split, y_test_split = y_train[train_index], y_train[test_index]
#     X_train_split = np.reshape(X_train_split, (X_train_split.shape[0],X_train_split.shape[1],1))
#     hist_split = train_model(my_model, X_train_split, y_train_split, number_of_epochs, batch_size)
#     hist_list = hist_list.append(hist_split)
#
#
## Plot the loss per epoch.
#metric = "mean_absolute_error"
#
#x_axis = np.linspace(1,len(hist_list),len(hist_list))
#fig, axs = plt.subplots(1, 1, figsize=(10, 6))
#axs.plot(x_axis, hist_list['mean_absolute_error'], color = "blue")
#axs.set_title('Model Accuracy')
#axs.set_xlabel('Epoch')
#axs.set_ylabel('Accuracy')
#axs.legend(['Training set', 'Validation set'])
#axs.grid(True)
#fig.show()
#
#my_model.save("Load_Prediction/LSTM/Training_Set_Size_Analysis/Models/DMST_LSTM_34L_Training_Set.h5")

# Load one of the models from the directory "Models".
my_model = keras.models.load_model("Load_Prediction/LSTM/Training_Set_Size_Analysis/Models/DMST_LSTM_1L_Training_Set.h5")
#my_model = keras.models.load_model("Load_Prediction/LSTM/Training_Set_Size_Analysis/Models/DMST_LSTM_12L_Training_Set.h5")
#my_model = keras.models.load_model("Load_Prediction/LSTM/Training_Set_Size_Analysis/Models/DMST_LSTM_14L_Training_Set.h5")
#my_model = keras.models.load_model("Load_Prediction/LSTM/Training_Set_Size_Analysis/Models/DMST_LSTM_25L_Training_Set.h5")
#my_model = keras.models.load_model("Load_Prediction/LSTM/Training_Set_Size_Analysis/Models/DMST_LSTM_34L_Training_Set.h5")
#my_model = keras.models.load_model("Load_Prediction/LSTM/Training_Set_Size_Analysis/Models/DMST_LSTM_35L_Training_Set.h5")

########################################################################################################################
# Predicting the electricity load.
########################################################################################################################

result_train = y_scaler.inverse_transform(my_model.predict(np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))))
result_train = result_train.reshape(-1,)
result_test = y_scaler.inverse_transform(my_model.predict(np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))))
result_test = result_test.reshape(-1,)

X_train = x_scaler.inverse_transform(X_train)
X_test = x_scaler.inverse_transform(X_test)
y_train = (y_scaler.inverse_transform(y_train)).reshape(-1,)
y_test = np.array(y_test.iloc[:,-1]).reshape(-1,)

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
# Save the results in a csv file.
########################################################################################################################

import csv
with open('Load_Prediction\LSTM\Training_Set_Size_Analysis\AF_34L.csv', 'w', newline='',) as file:
    writer = csv.writer(file)
    writer.writerow(["Method","MSE","MAE","RMSE"])
    writer.writerow(["34 L",
                     str(mean_squared_error(y_test,result_test)),
                     str(mean_absolute_error(y_test,result_test)),
                     str(np.sqrt(mean_squared_error(y_test,result_test)))
                     ])
