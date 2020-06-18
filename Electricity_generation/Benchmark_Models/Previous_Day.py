import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

########################################################################################################################

# Get data and data preprocessing.

########################################################################################################################

from Data_Entsoe.Data_Preprocessing.Get_Features_And_Label import return_features_and_labels

# Get the X (containing the features) and y (containing the labels) values
X, y = return_features_and_labels()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = False)

# Create the Previous Day
error_previousday_train = abs(y[:len(X_train),0] - X_train[:,0])
print("The mean absolute error from the previous day prediction on the training set is %.2f" %np.average(error_previousday_train))
print("The mean squarred error from the previous day prediction on the training set is %.2f" %np.average(error_previousday_train*error_previousday_train))

error_previousday_test = abs(y[-len(X_test):,0] - X_test[:,0])
print("The mean absolute error from the previous day prediction on the testing set is %.2f" %np.average(error_previousday_test))
print("The mean squarred error from the previous day prediction on the testing set is %.2f" %np.average(error_previousday_test*error_previousday_test))

########################################################################################################################

# Visualising the results

########################################################################################################################

fig, axes = plt.subplots(3)
axes[0].plot(y, linewidth=0.5)
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Actual ")
plt.show()

# Plot the predicted generation (from the day before) against the recording date.
axes[1].plot(X[:, 0], linewidth=0.5)
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Previous day prediction")
plt.show()

# Plot the error between the previous day and the actual generation.
axes[2].plot(error_previousday_test, linewidth=0.5)
axes[2].set_xlabel("Time")
axes[2].set_ylabel("Error previous day on test set.")
plt.show()

########################################################################################################################

# Save the results in a csv file.

########################################################################################################################

import csv
with open('/Users/benoitputzeys/PycharmProjects/NN-Predicitons/Compare_Models/Previous_Day_result.csv', 'w', newline='',) as file:
    writer = csv.writer(file)
    writer.writerow(["Method","MSE","MAE"])
    writer.writerow(["Previous_Day",str(np.mean(error_previousday_test*error_previousday_test)),str(np.mean(abs(error_previousday_test)))])
