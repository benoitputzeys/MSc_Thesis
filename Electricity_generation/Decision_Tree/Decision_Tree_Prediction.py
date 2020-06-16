from sklearn.tree import DecisionTreeRegressor
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

########################################################################################################################

# Get data and data preprocessing.

########################################################################################################################

from Data_Preprocessing.get_features_and_label import return_features_and_labels

# Get the X (containing the features) and y (containing the labels) values
X, y = return_features_and_labels()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = False)
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

#print(intermediate_result)
result_test = y_scaler.inverse_transform(regressor.predict(X_test))
result_train = y_scaler.inverse_transform(regressor.predict(X_train))

#print(result)
result_test = result_test.reshape((len(result_test), 1))
result_train = result_train.reshape((len(result_train), 1))

########################################################################################################################

# Visualising the results

########################################################################################################################

X_vals = []
for i in range(len(X)):
    X_vals = np.append(X_vals, i)
X_vals = np.reshape(X_vals,(len(X_vals),1))

figure1 = plt.figure(1)
plt.plot(X_vals, y, color = 'red')
plt.title('Decision Tree')
plt.xlabel('Settlement Period')
plt.ylabel('Actual Value')

fig, ax = plt.subplots(3)

ax[0].plot(X_vals[:len(X_train)], X_train_unscaled[:,0], color = 'blue')
ax[0].set_xlabel('Settlement Period')
ax[0].set_ylabel('Actual Train')

ax[1].plot(X_vals[:len(result_train)], result_train, color = 'blue')
ax[1].set_xlabel('Settlement Period')
ax[1].set_ylabel('Train Prediction')

error_train = result_train - y_scaler.inverse_transform(y_train)
print("The mean absolute error of the training set is %0.2f" %np.mean(abs(error_train)))
print("The mean squarred error of the training set is %0.2f" %np.mean(error_train*error_train))

ax[2].plot(abs(error_train), color = 'blue')
ax[2].set_xlabel('Settlement Period')
ax[2].set_ylabel('Train Error')
plt.show()

fig2, ax2 = plt.subplots(3)

ax2[0].plot(X_vals[-len(X_test):], X_test_unscaled[:,0], color = 'blue')
ax2[0].set_xlabel('Settlement Period')
ax2[0].set_ylabel('Actual Test')

ax2[1].plot(X_vals[len(result_train):], result_test, color = 'blue')
ax2[1].set_xlabel('Settlement Period')
ax2[1].set_ylabel('Test Prediction')

error_test = result_test - y_scaler.inverse_transform(y_test)
print("The mean absolute error of the test set is %0.2f" %np.mean(abs(error_test)))
print("The mean squarred error of the test set is %0.2f" %np.mean(error_test*error_test))

ax2[2].plot(abs(error_test), color = 'blue')
ax2[2].set_xlabel('Settlement Period')
ax2[2].set_ylabel('Test Error')
plt.show()


########################################################################################################################

# Save the results in a csv file.

########################################################################################################################

import csv
with open('/Users/benoitputzeys/PycharmProjects/NN-Predicitons/Compare_Models/Decision_Tree_result.csv', 'w', newline='',) as file:
    writer = csv.writer(file)
    writer.writerow(["Method","MSE","MAE"])
    writer.writerow(["Decision_Tree",str(np.mean(error_test*error_test)),str(np.mean(abs(error_test)))])