from matplotlib import pylab as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import matplotlib.ticker as plticker

# Get the X (containing the features) and y (containing the labels) values
X = pd.read_csv('Data_Preprocessing/For_1_SP_Step_Prediction/X.csv', delimiter=',')
X = X.set_index("Time")
dates = X.iloc[:,-1]
X = X.iloc[:,:-1]

y = pd.read_csv('Data_Preprocessing/For_1_SP_Step_Prediction/y.csv', delimiter=',')
y = y.set_index("Time")

# Split data into train set and test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle = False)

# Only take half the training set.
X_train = X_train[int(len(X_train)*1/2):]/1000
X_test = X_test[:int(len(X_test)*1/2)]/1000
y_train = y_train[int(len(y_train)*1/2):]/1000
y_test = y_test[:int(len(y_test)*1/2)]/1000
dates = dates[-len(X_train)-len(X_test)*2:-len(X_test)]

########################################################################################################################
# Create the model.
########################################################################################################################

# As the x-values are shifted by 1 SP from the past to the future, the prediction is already contained in the x-values.
pred_train = X_train.iloc[:,0]
pred_test = X_test.iloc[:,0]

########################################################################################################################
# Data processing for plotting curves and printing the errors.
########################################################################################################################

# Print the errors
print("-"*200)
print("The mean absolute error of the training set is %0.2f" % mean_absolute_error(y_train.iloc[:,0],pred_train))
print("The mean squared error of the training set is %0.2f" % mean_squared_error(y_train.iloc[:,0],pred_train))
print("The root mean squared error of the training set is %0.2f" % np.sqrt(mean_squared_error(y_train.iloc[:,0],pred_train)))

print("-"*200)
print("The mean absolute error of the training set is %0.2f" % mean_absolute_error(y_test.iloc[:,0],pred_test))
print("The mean squared error of the training set is %0.2f" % mean_squared_error(y_test.iloc[:,0],pred_test))
print("The root mean squared error of the training set is %0.2f" % np.sqrt(mean_squared_error(y_test.iloc[:,0],pred_test)))
print("-"*200)

########################################################################################################################
# Visualising the results
########################################################################################################################

# Create a vector that contains the error between the prediction and the test set values.
error_test_plot = np.zeros((336+48*3,1))
error_test_plot[-336:,0] = X_test.iloc[:48*7,0]-y_test.iloc[:48*7,0]

# Plot the result with the truth in blue/black and the predictions in orange.
fig2, axs2=plt.subplots(2,1,figsize=(12,6))
axs2[0].grid(True)
axs2[0].plot(dates.iloc[-len(X_test)-48*3:-len(X_test)],
             y_train.iloc[-48*3:,0],
             label = "Training set ", alpha = 1, color = "blue")
axs2[0].plot(dates.iloc[-len(X_test):-len(X_test)+48*7],
             pred_test[:48*7],
             label = "Naive prediction", color = "orange")
axs2[0].plot(dates.iloc[-len(X_test):-len(X_test)+48*7],
             y_test.iloc[:48*7,0],
             label = "Test set", alpha = 1, color = "black")
axs2[0].axvline(dates.iloc[-len(X_test)], linestyle="--", color = "black")
axs2[0].set_ylabel('Load, GW',size = 14)
axs2[0].plot(30,30, label = "Error", color = "red")

# plot the errors
axs2[1].grid(True)
axs2[1].plot(dates.iloc[-len(X_test)-48*3:-len(X_test)+48*7],
             error_test_plot,
             label = "Error", alpha = 1, color = "red")
axs2[1].axvline(dates.iloc[-len(X_test)], linestyle="--", color = "black")
axs2[1].set_xlabel('Date',size = 14)

# Include additional details such as tick intervals, rotation, legend positioning and grid on.
loc = plticker.MultipleLocator(base=47) # this locator puts ticks at regular intervals
axs2[0].xaxis.set_major_locator(loc), axs2[1].xaxis.set_major_locator(loc)
fig2.autofmt_xdate(rotation=15)
axs2[0].legend(loc=(1.02,0.6))
fig2.show()

########################################################################################################################
# Save the results in a csv file.
########################################################################################################################

df_errors = pd.DataFrame({"MSE_Train": [mean_squared_error(y_train.iloc[:,0],pred_train)],
                          "MAE_Train": [mean_absolute_error(y_train.iloc[:,0],pred_train)],
                          "RMSE_Train": [np.sqrt(mean_squared_error(y_train.iloc[:,0],pred_train))],
                          "MSE_Test": [mean_squared_error(y_test.iloc[:,0],pred_test)],
                          "MAE_Test": [mean_absolute_error(y_test.iloc[:,0],pred_test)],
                          "RMSE_Test": [np.sqrt(mean_squared_error(y_test.iloc[:,0],pred_test))],
                          })

df_errors.to_csv("Compare_Models/Single_Step_Results/Previous_SP_result.csv")

