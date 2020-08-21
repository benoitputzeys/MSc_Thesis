from sklearn.svm import SVR
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
from pandas import DataFrame
import matplotlib.ticker as plticker
import time

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

X_train = X_train[int(len(X_train)*1/2):]
X_test = X_test[:int(len(X_test)*1/2)]
y_train = y_train[int(len(y_train)*1/2):]
y_test = y_test[:int(len(y_test)*1/2)]
dates = dates[-len(X_train)-len(X_test)*2:-len(X_test)]

X_train_unscaled = X_train
X_test_unscaled = X_test

# Feature Scaling
x_scaler = StandardScaler()
y_scaler = StandardScaler()
X_train = x_scaler.fit_transform(X_train)
X_test = x_scaler.transform(X_test)
y_train = y_scaler.fit_transform(y_train)

########################################################################################################################
# Create the model.
########################################################################################################################

# Fit the SVR to our data
# Define the three parameters for the SVR: C, Epsilon and Gamme
# With gamma = 'scale', SVR uses 1 / (n_features * X.var()) as value of gamma
regressor = SVR(kernel = 'rbf', C = 1.0, epsilon = 0.1, gamma = 'scale')

# Measure the time to train the model.
start_time = time.time()
regressor.fit(X_train, y_train)
elapsed_time = time.time() - start_time

########################################################################################################################
# Predicting the generation on the test set and inverse the scaling.
########################################################################################################################

pred_train = y_scaler.inverse_transform(regressor.predict(X_train))/1000
pred_test = y_scaler.inverse_transform(regressor.predict(X_test))/1000

X_train = x_scaler.inverse_transform(X_train)
X_train[:,0] = X_train[:,0]/1000
X_test = x_scaler.inverse_transform(X_test)
X_test[:,0] = X_test[:,0]/1000
y_train = (y_scaler.inverse_transform(y_train)/1000).reshape(-1,)
y_test = np.array(y_test.iloc[:,-1]/1000)

########################################################################################################################
# Compute and print the errors.
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
# Visualising the results
########################################################################################################################

error_test_plot = np.zeros((48*3+48*7,1))
error_test_plot[-336:] = error_test[:48*7].reshape(-1,1)

# Plot the result with the truth in black/blue and the predictions in orange.
fig2, axs2=plt.subplots(2,1,figsize=(12,6))

# First plot contains the prediction and the true values from the test and training set.
axs2[0].plot(dates.iloc[-len(X_test)-48*3:-len(X_test)],
             y_train[-48*3:],
             label = "Training Set", alpha = 1, color = "blue")
axs2[0].plot(dates.iloc[-len(X_test):-len(X_test)+48*7],
             pred_test[:48*7],
             label = "SVR Pred.", color = "orange")
axs2[0].plot(dates.iloc[-len(X_test):-len(X_test)+48*7],
             y_test[:48*7],
             label = "Test Set", alpha = 1, color = "black")
axs2[0].axvline(dates.iloc[-len(X_test)], linestyle="--", color = "black")
axs2[0].set_ylabel('Load, GW',size = 14)
axs2[0].plot(30,30, label = "Error", color = "red")

# Second plot contains the errors.
axs2[1].plot(dates.iloc[-len(X_test)-48*3:-len(X_test)+48*7],
             error_test_plot,
             label = "Error", alpha = 1, color = "red")
axs2[1].axvline(dates.iloc[-len(X_test)], linestyle="--", color = "black")
axs2[1].set_xlabel('Date',size = 14)
axs2[1].set_ylabel('Error, GW',size = 14)

# Include additional details such as tick intervals, rotation, legend positioning and grid on.
axs2[0].grid(True), axs2[1].grid(True)
loc = plticker.MultipleLocator(base=47) # this locator puts ticks at regular intervals
axs2[1].xaxis.set_major_locator(loc), axs2[0].xaxis.set_major_locator(loc)
fig2.autofmt_xdate(rotation=0)
plt.xticks(np.arange(1,482, 48), ["14:00\n07/22","14:00\n07/23","14:00\n07/24",
                                  "14:00\n07/25","14:00\n07/26","14:00\n07/27",
                                  "14:00\n07/28","14:00\n07/29","14:00\n07/30",
                                  "14:00\n07/31","14:00\n08/01"])
axs2[0].legend(loc=(1.02,0.62))
fig2.show()
fig2.savefig("Load_Prediction/SVR/Figures/DMST_Prediction.pdf", bbox_inches='tight')

########################################################################################################################
# Save the results in a csv file.
########################################################################################################################

pd.DataFrame({"SVR_Time": [elapsed_time]}).to_csv("Compare_Models/Direct_Multi_Step_Probability_Results/Time_to_Train/SVR.csv")

df_errors = pd.DataFrame({"MSE_Train": [mean_squared_error(y_train,pred_train)],
                          "MAE_Train": [mean_absolute_error(y_train,pred_train)],
                          "RMSE_Train": [np.sqrt(mean_squared_error(y_train,pred_train))],
                          "MSE_Test": [mean_squared_error(y_test, pred_test)],
                          "MAE_Test": [mean_absolute_error(y_test, pred_test)],
                          "RMSE_Test": [np.sqrt(mean_squared_error(y_test, pred_test))],
                          })
df_errors.to_csv("Compare_Models/Direct_Multi_Step_Probability_Results/Probability_Based_on_Training/SVR_error.csv")
df_errors.to_csv("Compare_Models/Direct_Multi_Step_Results/SVR.csv")

pd.DataFrame({"Test_Prediction":pred_test}).to_csv("Load_Prediction/SVR/Direct_Multi_Step_Prediction/Pred_Test.csv")
pd.DataFrame({"Train_Prediction":pred_train}).to_csv("Load_Prediction/SVR/Direct_Multi_Step_Prediction/Pred_Train.csv")

