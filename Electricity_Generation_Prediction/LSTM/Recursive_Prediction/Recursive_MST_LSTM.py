import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from Electricity_Generation_Prediction.LSTM.Functions_LSTM import plot_the_loss_curve, train_model, create_model, plot_generation, plot_prediction_zoomed_in
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.ticker as plticker
import keras
from pandas import DataFrame

########################################################################################################################
# Get data and data preprocessing.
########################################################################################################################

# Get the X (containing the features) and y (containing the labels) values
X = pd.read_csv('Data_Preprocessing/For_Single_Step_Prediction/X.csv', delimiter=',')
X = X.set_index("Time")
X = X.drop(columns = "Transmission_Past")
dates = X.iloc[:,-1]
X = X.iloc[:,:-5]

y = pd.read_csv('Data_Preprocessing/For_Single_Step_Prediction/y.csv', delimiter=',')
y = y.set_index("Time")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = False)

X_train = X_train[int(len(X_train)*1/2):]
y_train = y_train[int(len(y_train)*1/2):]
dates = dates[-len(X_train)-len(X_test):]

X_test_unscaled = X_test
X_train_unscaled = X_train

# Feature Scaling
x_scaler = StandardScaler()
y_scaler = StandardScaler()
X_train = x_scaler.fit_transform(X_train)
X_test = x_scaler.transform(X_test)
y_train = y_scaler.fit_transform(y_train)

########################################################################################################################
# Create the model.
########################################################################################################################

my_model = keras.models.load_model("Electricity_Generation_Prediction/LSTM/Recursive_Prediction/SST_No_Trans_No_Date.h5")

########################################################################################################################
# Predicting the generation.
########################################################################################################################

# Multi-Step prediction
X_future_features = pd.DataFrame(data=X_train_unscaled.iloc[-49:,:].values,  columns=["0","1","2","3","4","5"])
result_future = y_scaler.inverse_transform(y_train[-2:])

for i in range(0,48*7):

    prev_value = result_future[-2]
    new_row = [[prev_value[0], 0, 0, 0, 0, 0]]
    new_row = DataFrame(new_row, columns=["0","1","2","3","4","5"])

    X_future_features = pd.concat([X_future_features,new_row])
    rolling_mean_10 = X_future_features["0"].rolling(window=10).mean().values[-1]
    rolling_mean_50 = X_future_features["0"].rolling(window=50).mean().values[-1]
    exp_20 = X_future_features["0"].ewm(span=20, adjust=False).mean().values[-1]
    exp_50 = X_future_features["0"].ewm(span=50, adjust=False).mean().values[-1]

    update_row = [[prev_value,
                   rolling_mean_10,
                   rolling_mean_50,
                   exp_20, exp_50,
                   X_test_unscaled.iloc[i,5]
                   ]]

    update_row = DataFrame(update_row, columns=["0","1","2","3","4","5"])
    X_future_features.iloc[-1,:] = update_row.iloc[0,:]

    result_future = np.append(result_future, y_scaler.inverse_transform(my_model.predict(x_scaler.transform(update_row).reshape(1,6,1))))
    result_future = np.reshape(result_future,(-1,1))

result_future = result_future[2:]

########################################################################################################################
# Inverse the scaling.
########################################################################################################################

X_train = x_scaler.inverse_transform(X_train)
X_test = x_scaler.inverse_transform(X_test)
y_train = y_scaler.inverse_transform(y_train)

########################################################################################################################
# Data processing for plotting curves and printing the errors.
########################################################################################################################

print("-"*200)
error_test = abs(result_future - y_test[:48*7])
print("The mean absolute error of the prediction is %0.2f" % mean_absolute_error(y_test[:48*7],result_future[-48*7:]))
print("The mean squared error of the prediction is %0.2f" % mean_squared_error(y_test[:48*7],result_future[-48*7:]))
print("The root mean squared error of prediction set is %0.2f" % np.sqrt(mean_squared_error(y_test[:48*7],result_future[-48*7:])))
print("-"*200)

########################################################################################################################
# Plotting curves.
########################################################################################################################

error_test_plot = np.zeros((48*3+48*7,1))
error_test_plot[-336:] = error_test[:48*7]
# Plot the result with the truth in red and the predictions in blue.
fig2, axs2=plt.subplots(2,1,figsize=(12,6))
axs2[0].grid(True)
axs2[0].plot(dates.iloc[-len(X_test)-48*3:-len(X_test)],y_train[-48*3:,0]/1000, label = "Training Set (True Values)", alpha = 1, color = "black")
axs2[0].plot(dates.iloc[-len(X_test):-len(X_test)+48*7],result_future[:48*7,0]/1000, label = "LSTM Recursive Prediction", color = "orange")
axs2[0].plot(dates.iloc[-len(X_test):-len(X_test)+48*7],y_test[:48*7]/1000, label = "Test Set (True Values)", alpha = 1, color = "blue")
axs2[0].axvline(dates.iloc[-len(X_test)], linestyle="--", color = "black")
axs2[0].set_ylabel('Load [GW]',size = 14)
loc = plticker.MultipleLocator(base=47) # this locator puts ticks at regular intervals
axs2[0].xaxis.set_major_locator(loc)

axs2[1].grid(True)
axs2[1].plot(dates.iloc[-len(X_test)-48*3:-len(X_test)+48*7], error_test_plot/1000, label = "Absolute Error", alpha = 1, color = "red")
axs2[1].axvline(dates.iloc[-len(X_test)], linestyle="--", color = "black")
axs2[1].set_xlabel('Date',size = 14)
axs2[1].set_ylabel('Absolute Error [GW]',size = 14)
loc = plticker.MultipleLocator(base=47) # this locator puts ticks at regular intervals
axs2[1].xaxis.set_major_locator(loc)
fig2.autofmt_xdate(rotation=15)

axs2[1].legend(loc=(1.04,0.9))
axs2[0].legend(loc=(1.04,0.7))

fig2.show()

########################################################################################################################
# Save the results in a csv file.
########################################################################################################################

# Continue with this method MST_1 ????
# pd.DataFrame(result_future).to_csv("/Electricity_Generation_Prediction/Hybrid_Model/Pred_train2_other_metrics/LSTM_prediction.csv")

