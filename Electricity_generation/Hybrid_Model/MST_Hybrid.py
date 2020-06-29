import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from Electricity_generation.Hybrid_Model.Functions import create_model, train_model, plot_the_loss_curve
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import keras

########################################################################################################################
# Get data and data preprocessing.
########################################################################################################################

from numpy import genfromtxt

# Get the X (containing the features) and y (containing the labels) values
ANN = genfromtxt('/Users/benoitputzeys/PycharmProjects/MSc_Thesis/Electricity_generation/Hybrid_Model/ANN_prediction.csv', delimiter=',')
SVR = genfromtxt('/Users/benoitputzeys/PycharmProjects/MSc_Thesis/Electricity_generation/Hybrid_Model/SVR_prediction.csv', delimiter=',')
SARIMA = genfromtxt('/Users/benoitputzeys/PycharmProjects/MSc_Thesis/Electricity_generation/Hybrid_Model/SARIMA_prediction.csv', delimiter=',')

# Get the X (containing the features) and y (containing the labels) values
X = genfromtxt('/Users/benoitputzeys/PycharmProjects/MSc_Thesis/Data_Entsoe/Data_Preprocessing/X.csv', delimiter=',')
y = genfromtxt('/Users/benoitputzeys/PycharmProjects/MSc_Thesis/Data_Entsoe/Data_Preprocessing/y.csv', delimiter=',')
y = np.reshape(y, (len(y), 1))

# Split data into train set and test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle=False)

ANN = ANN[1:,1]
ANN = ANN.reshape(-1,1)
SVR = SVR[1:,1]
SVR = SVR.reshape(-1,1)
SARIMA = SARIMA[1:,1]
SARIMA = SARIMA.reshape(-1,1)
average = (ANN + SVR + SARIMA)/3
all_predictions = np.concatenate((ANN, SVR, SARIMA, average), axis = 1)

########################################################################################################################
# Plot average.
########################################################################################################################

plt.plot(all_predictions[:,-1])
plt.plot(y_test[:48*7])

########################################################################################################################
# Create the model.
########################################################################################################################

# Define the hyperparameters.
learning_rate = 0.001
number_of_epochs = 50
batch_size = 32

# Create the model.
my_model = create_model(3, learning_rate)

# Extract the loss per epoch to plot the learning progress.

hist_list = pd.DataFrame()

hist_split = train_model(my_model, all_predictions[:,0:3], y_test[0:48*7], number_of_epochs, batch_size)
hist_list = hist_list.append(hist_split)

# Plot the loss per epoch.
metric = "mean_absolute_error"
plot_the_loss_curve(np.linspace(1,len(hist_list), len(hist_list) ), hist_list[metric], metric)

########################################################################################################################
# Make predictions and compute the errors.
########################################################################################################################

result_test = my_model.predict(all_predictions[:,0:3])

error = result_test-y_test[:48*7]
# Get the errors.
print("-"*200)
print("The mean absolute error of the test set is %0.2f" % np.average(abs(error)))
print("The mean squared error of the test set is %0.2f" % np.average(abs(error)**2))
print("The root mean squared error of the test set is %0.2f" % np.sqrt(np.mean(abs(error)**2)))
print("The mean absolute percent error of the test set is %0.2f" % np.mean(abs((y_test[:48*7]-result_test)/y_test[:48*7])))
print("-"*200)

########################################################################################################################
# Make predictions and compute the errors.
########################################################################################################################

fig, axes = plt.subplots(3)
axes[0].plot(result_test, linewidth=0.5, label ="Prediction 7 days in the future with hybrid")
axes[0].set_xlabel("Settlement Period")
axes[0].set_ylabel("Electricity Load [MW]")
#y_values_dates = create_dates(X_future_features[-48*7:].to_numpy(), y_scaler.inverse_transform(y_test[:48*7]))
axes[0].plot(y_test[0:48*7],linewidth=0.5, label="Actual")
axes[0].legend()

axes[1].plot(all_predictions[:,-1], linewidth=0.5, label ="Prediction 7 days in the future with average hybrid")
axes[1].plot(y_test[0:48*7],linewidth=0.5, label="Actual")
axes[1].set_xlabel("Settlement Period")
axes[1].set_ylabel("Electricity Load [MW]")
axes[1].legend()

axes[2].plot(abs(error), linewidth=0.5, label ="Error hybrid")
axes[2].set_xlabel("Settlement Period")
axes[2].set_ylabel("Electricity Load [MW]")
axes[2].legend()
