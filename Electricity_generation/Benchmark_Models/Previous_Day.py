import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import datetime
from pandas import DataFrame

def create_dates(features_df, y_values):

    date_list = [datetime.datetime(year=int(features_df[i, -1]),
                                   month=int(features_df[i, -2]),
                                   day=int(features_df[i, -3]),
                                   hour=int((features_df[i, -4] - 1) / 2),
                                   minute=(i % 2) * 30) for i in range(len(features_df))]
    df_dates = DataFrame(date_list, columns=['Date'])
    df_dates = df_dates.set_index(['Date'])
    df_dates['Load'] = y_values

    return df_dates

########################################################################################################################
# Get data and data preprocessing.
########################################################################################################################

from numpy import genfromtxt

# Get the X (containing the features) and y (containing the labels) values
X = genfromtxt('/Users/benoitputzeys/PycharmProjects/MSc_Thesis/Data_Entsoe/Data_Preprocessing/X.csv', delimiter=',')
y = genfromtxt('/Users/benoitputzeys/PycharmProjects/MSc_Thesis/Data_Entsoe/Data_Preprocessing/y.csv', delimiter=',')
y = np.reshape(y, (len(y), 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = False)

# Create the Previous Day
print("-"*200)
error_train = y[:len(X_train),0] - X_train[:,0]
print("The mean absolute error of the training set is %0.2f" % mean_absolute_error(y[:len(X_train),0],X_train[:,0]))
print("The mean squared error of the training set is %0.2f" % mean_squared_error(y[:len(X_train),0],X_train[:,0]))
print("The root mean squared error of the training set is %0.2f" % np.sqrt(mean_squared_error(y[:len(X_train),0],X_train[:,0])))

print("-"*200)
error_test = y[-len(X_test):,0] - X_test[:,0]
print("The mean absolute error of the training set is %0.2f" % mean_absolute_error(y[-len(X_test):,0],X_test[:,0]))
print("The mean squared error of the training set is %0.2f" % mean_squared_error(y[-len(X_test):,0],X_test[:,0]))
print("The root mean squared error of the training set is %0.2f" % np.sqrt(mean_squared_error(y[-len(X_test):,0],X_test[:,0])))

########################################################################################################################
# Visualising the results
########################################################################################################################

y_values_dates = create_dates(X, y)
fig, axes = plt.subplots(3)
fig.suptitle('Training + Test Set (Previous Day)', fontsize=16)
axes[0].plot(y_values_dates, linewidth=0.5)
axes[0].set_xlabel("Settlement Period")
axes[0].set_ylabel("Actual")
plt.show()

y_values_dates = create_dates(X, X[:, 0])
# Plot the predicted generation (from the day before) against the recording date.
axes[1].plot(y_values_dates, linewidth=0.5)
axes[1].set_xlabel("Settlement Period")
axes[1].set_ylabel("Previous day prediction")
plt.show()

y_values_dates = create_dates(X, np.append(error_train,error_test))
# Plot the error between the previous day and the actual generation.
axes[2].plot(y_values_dates, linewidth=0.5)
axes[2].set_xlabel("Settlement Period")
axes[2].set_ylabel("Absolute error on test set.")
plt.show()

########################################################################################################################
# Save the results in a csv file.
########################################################################################################################

import csv
with open('/Users/benoitputzeys/PycharmProjects/MSc_Thesis/Compare_Models/SST_results/Previous_Day_result.csv', 'w', newline='', ) as file:
    writer = csv.writer(file)
    writer.writerow(["Method","MSE","MAE","RMSE"])
    writer.writerow(["Previous_Day",
                     str(mean_squared_error(y[-len(X_test):,0],X_test[:,0])),
                     str(mean_absolute_error(y[-len(X_test):,0],X_test[:,0])),
                     str(np.sqrt(mean_squared_error(y[-len(X_test):,0],X_test[:,0])))
                     ])
