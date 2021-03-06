import numpy as np
import matplotlib.pyplot as plt
from Load_Prediction.LSTM.Functions_LSTM import train_model, create_model
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
X = X.iloc[:,:-1]
y = pd.read_csv('Data_Preprocessing/For_336_SP_Step_Prediction/y.csv', delimiter=',')
y = y.set_index("Time")

# Partition the data into training set (80%) and test set (20%).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = False)

# Save the unscaled data for later for data representation.
X_test_unscaled = X_test
X_train_unscaled = X_train

# Only take half the training data and half the test data.
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

##Different models have to be used to visualise the impact of different features.
#
## Define the hyperparameters.
#learning_rate = 0.005
#number_of_epochs = 40
#batch_size = 19
#
## Create the model.
#my_model = create_model(X_train, learning_rate)
#
## Extract the loss per epoch to plot the learning progress.
#hist_list = pd.DataFrame()
#
## Perfirm 5-fold cross validation
#tscv = TimeSeriesSplit()
#for train_index, test_index in tscv.split(X_train):
#     X_train_split, X_test_split = X_train[train_index], X_train[test_index]
#     y_train_split, y_test_split = y_train[train_index], y_train[test_index]
#     X_train_split = np.reshape(X_train_split, (X_train_split.shape[0],X_train_split.shape[1],1))
#     hist_split = train_model(my_model, X_train_split, y_train_split, number_of_epochs, batch_size)
#     hist_list = hist_list.append(hist_split)
#
## Plot the loss per epoch.
#metric = "mean_absolute_error"
#x_axis = np.linspace(1,len(hist_list),len(hist_list))
#
#fig, axs = plt.subplots(1, 1, figsize=(10, 6))
#axs.plot(x_axis, hist_list['mean_absolute_error'], color = "blue")
#axs.set_xlabel('Epoch')
#axs.set_ylabel('Loss')
#axs.legend(['Training set'])
#axs.grid(True)
#fig.show()
#
## Save the model.
#my_model.save("Load_Prediction/LSTM/Feature_Analysis/Models/DMST_LSTM_F7_SP_DoW_D_M_Y.h5")


#Different models have to be used to visualise the impact of the different features.
#Here are all the trained models to analyse the impact of leaving Date-related features out.
#DMST_LSTM_F7 means all 7 features are used DMST_LSTM_F7_SP means all 7 features are used PLUS the SP as well
#DMST_LSTM_F7_SP_DoW means all 7 features PLUS the SP PLUS the Day of the Week

#my_model = keras.models.load_model("Load_Prediction/LSTM/Feature_Analysis/Models/DMST_LSTM_F7.h5")
#my_model = keras.models.load_model("Load_Prediction/LSTM/Feature_Analysis/Models/DMST_LSTM_F7_SP.h5")
#my_model = keras.models.load_model("Load_Prediction/LSTM/Feature_Analysis/Models/DMST_LSTM_F7_SP_DoW.h5")
#my_model = keras.models.load_model("Load_Prediction/LSTM/Feature_Analysis/Models/DMST_LSTM_F7_SP_DoW_D.h5")
#my_model = keras.models.load_model("Load_Prediction/LSTM/Feature_Analysis/Models/DMST_LSTM_F7_SP_DoW_D_M.h5")
my_model = keras.models.load_model("Load_Prediction/LSTM/Feature_Analysis/Models/DMST_LSTM_F7_SP_DoW_D_M_Y.h5")


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

# Compute the error between the actual generation and the prediction from the LSTM
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
# Save the results in a csv file.
########################################################################################################################

import csv
with open('Load_Prediction/LSTM/Feature_Analysis/F7_SP_DoW_D_M_Y.csv', 'w', newline='',) as file:
    writer = csv.writer(file)
    writer.writerow(["Method","MSE","MAE","RMSE"])
    writer.writerow(["F7_SP_DoW_D_M_Y",
                     str(mean_squared_error(y_test,result_test)),
                     str(mean_absolute_error(y_test,result_test)),
                     str(np.sqrt(mean_squared_error(y_test,result_test)))
                     ])

