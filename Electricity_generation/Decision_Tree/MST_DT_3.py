from sklearn.tree import DecisionTreeRegressor
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
from pandas import DataFrame

########################################################################################################################
# Get data and data preprocessing.
########################################################################################################################

from numpy import genfromtxt

# Get the X (containing the features) and y (containing the labels) values
X = genfromtxt('Data_Entsoe/Data_Preprocessing/X.csv', delimiter=',')
y = genfromtxt('Data_Entsoe/Data_Preprocessing/y.csv', delimiter=',')
y = np.reshape(y, (len(y), 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = False)
# Save the unscaled data for later for data representation.
X_test_unscaled = X_test
X_train_unscaled = X_train

# Feature Scaling
x_scaler = StandardScaler()
y_scaler = StandardScaler()
X_train = x_scaler.fit_transform(X_train)
X_test = x_scaler.transform(X_test)
y_train = y_scaler.fit_transform(y_train)
y_test = y_scaler.transform(y_test)

########################################################################################################################
# Create the model.
########################################################################################################################

# Fit the Decision Tree to our data
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)

# Single-Step prediction
result_train = y_scaler.inverse_transform(regressor.predict(X_train))
result_test = y_scaler.inverse_transform(regressor.predict(X_test))

result_test = result_test.reshape((len(result_test), 1))
result_train = result_train.reshape((len(result_train), 1))

# Multi-Step prediction

X_future_features = pd.DataFrame(data=X_test_unscaled,  columns=["0","1","2","3","4","5"])
DoW_SP = genfromtxt('Data_Entsoe/Data_Preprocessing/For_Multi_Step_Prediction/DoW_SP_2.csv',delimiter=',')

result_future = y_scaler.inverse_transform(y_test)

for i in range(0,48*7):

    prev_value = result_future[-1]

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

    result_future = np.append(result_future, y_scaler.inverse_transform(regressor.predict(x_scaler.transform(update_row))))
    result_future = np.reshape(result_future,(-1,1))

print("-"*200)
error_train = result_train - y_scaler.inverse_transform(y_train)
print("The mean absolute error of the training set is %0.2f" % mean_absolute_error(y_scaler.inverse_transform(y_train),result_train))
print("The mean squared error of the training set is %0.2f" % mean_squared_error(y_scaler.inverse_transform(y_train),result_train))
print("The root mean squared error of the training set is %0.2f" % np.sqrt(mean_squared_error(y_scaler.inverse_transform(y_train),result_train)))

print("-"*200)
error_test = result_test - y_scaler.inverse_transform(y_test)
print("The mean absolute error of the training set is %0.2f" % mean_absolute_error(y_scaler.inverse_transform(y_test),result_test))
print("The mean squared error of the training set is %0.2f" % mean_squared_error(y_scaler.inverse_transform(y_test),result_test))
print("The root mean squared error of the training set is %0.2f" % np.sqrt(mean_squared_error(y_scaler.inverse_transform(y_test),result_test)))
print("-"*200)

########################################################################################################################
# Visualising the results
########################################################################################################################

figure1 = plt.figure(1)
plt.plot(y, linewidth=0.5)
plt.title('Training + Test Set (Decision Tree)')
plt.xlabel('Settlement Period')
plt.ylabel('Actual Value (Training + Test Set)')

fig, ax = plt.subplots(3)
fig.suptitle('Decision Tree: Training Set', fontsize=16)
ax[0].plot(X_train_unscaled[:,0],linewidth=0.5)
ax[0].set_xlabel('Settlement Period')
ax[0].set_ylabel('Actual Value: Training Set')

ax[1].plot(result_train, linewidth=0.5)
ax[1].set_xlabel('Settlement Period')
ax[1].set_ylabel('Single-Step Prediction on training set')

ax[2].plot(abs(error_train), linewidth=0.5)
ax[2].set_xlabel('Settlement Period')
ax[2].set_ylabel('Absolute error: Training set')
plt.show()

fig2, ax2 = plt.subplots(3)
fig2.suptitle('Decision Tree: Testing Set', fontsize=16)
ax2[0].plot(X_test_unscaled[:,0], linewidth=0.5)
ax2[0].set_xlabel('Settlement Period')
ax2[0].set_ylabel('Single-Step Actual Value: Test Set')

ax2[1].plot(result_test[:-48*7],linewidth=0.5)
ax2[1].set_xlabel('Settlement Period')
ax2[1].set_ylabel('Prediction on test set')

ax2[2].plot(abs(error_test), linewidth=0.5)
ax2[2].set_xlabel('Settlement Period')
ax2[2].set_ylabel('Absolute error: Test set.')
plt.show()

figure1 = plt.figure(4)
plt.plot(y_scaler.inverse_transform(result_future)[-48*7:], linewidth=0.5)
plt.title('Prediction 7 days in the future')
plt.xlabel('Settlement Period')
plt.ylabel('Prediction')

values = X_future_features-X_future_features.shift(-1)

########################################################################################################################
# Save the results in a csv file.
########################################################################################################################

import csv
with open('Compare_Models/SST_results/Decision_Tree_result.csv', 'w', newline='', ) as file:
    writer = csv.writer(file)
    writer.writerow(["Method","MSE","MAE","RMSE"])
    writer.writerow(["Decision_Tree",
                     str(mean_squared_error(y_scaler.inverse_transform(y_test),result_test)),
                     str(mean_absolute_error(y_scaler.inverse_transform(y_test),result_test)),
                     str(np.sqrt(mean_squared_error(y_scaler.inverse_transform(y_test),result_test)))
                     ])