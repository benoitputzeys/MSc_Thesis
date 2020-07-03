import numpy as np
import matplotlib.pyplot as plt
from Electricity_generation.ANN.Functions_ANN import create_dates
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pandas import DataFrame
from tensorflow import keras

########################################################################################################################
# Get data and data preprocessing.
########################################################################################################################

from numpy import genfromtxt

# Get the X (containing the features) and y (containing the labels) values
X = genfromtxt('Data_Entsoe/Data_Preprocessing/X.csv', delimiter=',')
y = genfromtxt('Data_Entsoe/Data_Preprocessing/y.csv', delimiter=',')
y = np.reshape(y, (len(y), 1))

# Split data into train set and test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle=False)

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

my_model = keras.models.load_model("my_model.h5")

########################################################################################################################
# Predicting the generation.
########################################################################################################################

# Multi-Step prediction

X_future_features = pd.DataFrame(data=X_train_unscaled[-49:,:],  columns=["0","1","2","3","4","5","6","7","8","9"])
result_future = y_scaler.inverse_transform(y_train[-2:])

for i in range(0,48*7):

        prev_value = result_future[-2]
        new_row = [[prev_value[0], 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        new_row = DataFrame(new_row, columns=["0","1","2","3","4","5","6","7","8","9"])

        X_future_features = pd.concat([X_future_features,new_row])
        rolling_mean_10 = X_future_features["0"].rolling(window=10).mean().values[-1]
        rolling_mean_50 = X_future_features["0"].rolling(window=50).mean().values[-1]
        exp_20 = X_future_features["0"].ewm(span=20, adjust=False).mean().values[-1]
        exp_50 = X_future_features["0"].ewm(span=50, adjust=False).mean().values[-1]

        update_row = [[prev_value, rolling_mean_10, rolling_mean_50, exp_20, exp_50, X_test_unscaled[i,5], X_test_unscaled[i,6], X_test_unscaled[i,7], X_test_unscaled[i,8], X_test_unscaled[i,9]]]

        update_row = DataFrame(update_row, columns=["0","1","2","3","4","5","6","7","8","9"])
        X_future_features.iloc[-1,:] = update_row.iloc[0,:]

        result_future = np.append(result_future, y_scaler.inverse_transform(my_model.predict(x_scaler.transform(update_row))))
        result_future = np.reshape(result_future,(-1,1))

result_future = result_future[2:]

########################################################################################################################
# Data processing for plotting curves and printing the errors.
########################################################################################################################

print("-"*200)
error_test = abs(result_future - y_scaler.inverse_transform(y_test[:48*7]))
print("The mean absolute error of the prediction is %0.2f" % mean_absolute_error(y_scaler.inverse_transform(y_test[:48*7]),result_future[-48*7:]))
print("The mean squared error of the prediction is %0.2f" % mean_squared_error(y_scaler.inverse_transform(y_test[:48*7]),result_future[-48*7:]))
print("The root mean squared error of prediction set is %0.2f" % np.sqrt(mean_squared_error(y_scaler.inverse_transform(y_test[:48*7]),result_future[-48*7:])))
print("-"*200)

########################################################################################################################
# Plotting curves.
########################################################################################################################

figure1 = plt.figure(1)
plt.plot(create_dates(X_future_features[-48*7:].to_numpy(),result_future), linewidth=0.5)
plt.title('Prediction 7 days in the future with ANN')
plt.xlabel('Settlement Period')
plt.ylabel('Electricity Load [MW]')

fig2, axes2 = plt.subplots(2)
axes2[0].plot(create_dates(X_train_unscaled[-48*3:], y_scaler.inverse_transform(y_train[-48*3:])), linewidth=0.5, label ="Past load")
axes2[0].plot(create_dates(X_future_features[-48*7:].to_numpy(),result_future), linewidth=0.5, label ="Prediction 7 days in the future with LSTM")
axes2[0].set_xlabel("Settlement Period")
axes2[0].set_ylabel("Electricity Load [MW]")
axes2[0].plot(create_dates(X_future_features[-48*7:].to_numpy(), y_scaler.inverse_transform(y_test[:48*7])),linewidth=0.5, label="Actual")

fig2.legend()
axes2[1].plot(create_dates(X_future_features[-48*7:].to_numpy(),error_test),linewidth=0.5, label ="Absolute Error",color= "black")
axes2[1].set_xlabel("Settlement Period")
axes2[1].set_ylabel("Error in Prediction [MW]")
fig2.legend()

########################################################################################################################
# Save the results in a csv file.
########################################################################################################################

# Continue with this method MST_1 ????
# pd.DataFrame(result_future).to_csv("/Electricity_generation/Hybrid_Model/Pred_train2_other_metrics/ANN_prediction.csv")

