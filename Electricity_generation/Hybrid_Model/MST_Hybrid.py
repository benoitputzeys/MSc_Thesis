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
ANN_train = genfromtxt('Electricity_generation/Hybrid_Model/Pred_train2_other_metrics/ANN_prediction.csv', delimiter=',')
RF_train = genfromtxt('Electricity_generation/Hybrid_Model/Pred_train2_other_metrics/RF_prediction.csv', delimiter=',')
DT_train = genfromtxt('Electricity_generation/Hybrid_Model/Pred_train2_other_metrics/DT_prediction.csv', delimiter=',')
SVR_train = genfromtxt('Electricity_generation/Hybrid_Model/Pred_train2_other_metrics/SVR_prediction.csv', delimiter=',')
LSTM_train = genfromtxt('Electricity_generation/Hybrid_Model/Pred_train2_other_metrics/LSTM_prediction.csv', delimiter=',')
SARIMA_train = genfromtxt('Electricity_generation/Hybrid_Model/Pred_train2_other_metrics/SARIMA_prediction.csv', delimiter=',')

ANN_test = genfromtxt('Electricity_generation/Hybrid_Model/Pred_test_other_metrics/ANN_prediction.csv', delimiter=',')
RF_test = genfromtxt('Electricity_generation/Hybrid_Model/Pred_test_other_metrics/RF_prediction.csv', delimiter=',')
DT_test = genfromtxt('Electricity_generation/Hybrid_Model/Pred_test_other_metrics/DT_prediction.csv', delimiter=',')
SVR_test = genfromtxt('Electricity_generation/Hybrid_Model/Pred_test_other_metrics/SVR_prediction.csv', delimiter=',')
LSTM_test = genfromtxt('Electricity_generation/Hybrid_Model/Pred_test_other_metrics/LSTM_prediction.csv', delimiter=',')
SARIMA_test = genfromtxt('Electricity_generation/Hybrid_Model/Pred_test_other_metrics/SARIMA_prediction.csv', delimiter=',')

# Get the X (containing the features) and y (containing the labels) values
X = genfromtxt('Data_Entsoe/Data_Preprocessing/For_Multi_Step_Prediction/X.csv', delimiter=',')
y = genfromtxt('Data_Entsoe/Data_Preprocessing/For_Multi_Step_Prediction/y.csv', delimiter=',')
y = np.reshape(y, (len(y), 1))

# Save the unscaled data for later for data representation.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = False)
X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(X_train, y_train, test_size = 0.2, random_state = 0, shuffle = False)

ANN_train = ANN_train[1:,1]
ANN_train = ANN_train.reshape(-1,1)
ANN_test = ANN_test[1:,1]
ANN_test = ANN_test.reshape(-1,1)

DT_train = DT_train[1:,1]
DT_train = DT_train.reshape(-1,1)
DT_test = DT_test[1:,1]
DT_test = DT_test.reshape(-1,1)

RF_train = RF_train[1:,1]
RF_train = RF_train.reshape(-1,1)
RF_test = RF_test[1:,1]
RF_test = RF_test.reshape(-1,1)

LSTM_train = LSTM_train[1:,1]
LSTM_train = LSTM_train.reshape(-1,1)
LSTM_test = LSTM_test[1:,1]
LSTM_test = LSTM_test.reshape(-1,1)

SVR_train = SVR_train[1:,1]
SVR_train = SVR_train.reshape(-1,1)
SVR_test = SVR_test[1:,1]
SVR_test = SVR_test.reshape(-1,1)

SARIMA_train = SARIMA_train[1:,1]
SARIMA_train = SARIMA_train.reshape(-1,1)
SARIMA_test = SARIMA_test[1:,1]
SARIMA_test = SARIMA_test.reshape(-1,1)

average = (ANN_train + SVR_train + LSTM_train + DT_train + RF_train)/5
all_predictions_train = np.concatenate((ANN_train, SVR_train, LSTM_train, DT_train, RF_train), axis = 1)
all_predictions_average = np.concatenate((ANN_train, SVR_train, LSTM_train, DT_train, RF_train, average), axis = 1)
all_predictions_test = np.concatenate((ANN_test, SVR_test, LSTM_test, DT_test, RF_test), axis = 1)

# Feature Scaling
x_scaler = StandardScaler()
y_scaler = StandardScaler()
all_predictions_train = x_scaler.fit_transform(all_predictions_train)
all_predictions_test = x_scaler.transform(all_predictions_test)
y_train_2 = y_scaler.fit_transform(y_train_2)

########################################################################################################################
# Plot average.
########################################################################################################################

# plt.plot(all_predictions_average[:,-1])
# plt.plot(y_train_2[:48*7])

########################################################################################################################
# Create the model.
########################################################################################################################

# # Define the hyperparameters.
# learning_rate = 0.001
# number_of_epochs = 125
# batch_size = 32
#
# # Create the model.
# my_model = create_model(5, learning_rate)
#
# # Extract the loss per epoch to plot the learning progress.
#
# hist_list = pd.DataFrame()
#
# hist_split = train_model(my_model, all_predictions_train, y_train_2, number_of_epochs, batch_size)
# hist_list = hist_list.append(hist_split)
#
# # Plot the loss per epoch.
# metric = "mean_absolute_error"
# plot_the_loss_curve(np.linspace(1,len(hist_list), len(hist_list) ), hist_list[metric], metric)
#
# my_model.save("my_model.h5")

my_model = keras.models.load_model("my_model.h5")

########################################################################################################################
# Make predictions and compute the errors.
########################################################################################################################

result_train_2 = y_scaler.inverse_transform(my_model.predict(all_predictions_train))
result_test = y_scaler.inverse_transform(my_model.predict(all_predictions_test))
y_train_2 = y_scaler.inverse_transform(y_train_2)
error_train_2 = result_train_2-y_train_2
error_test = result_test - y_test

# Get the errors.
print("-"*200)
print("The mean absolute error of the train set 2 is %0.2f" % np.average(abs(error_train_2)))
print("The mean squared error of the train set 2 is %0.2f" % np.average(abs(error_train_2)**2))
print("The root mean squared error of the train set 2 is %0.2f" % np.sqrt(np.mean(abs(error_train_2)**2)))
print("The mean absolute percent error of the train set 2 is %0.2f" % np.mean(abs((y_train_2-result_train_2)/y_train_2)))
print("-"*200)

# Get the errors.
print("The mean absolute error of the test set is %0.2f" % np.average(abs(error_test)))
print("The mean squared error of the test set is %0.2f" % np.average(abs(error_test)**2))
print("The root mean squared error of the test set is %0.2f" % np.sqrt(np.mean(abs(error_test)**2)))
print("The mean absolute percent error of the test set is %0.2f" % np.mean(abs((y_test-result_test)/y_test)))
print("-"*200)

########################################################################################################################
# Make predictions and compute the errors.
########################################################################################################################

fig, axes = plt.subplots(2)
axes[0].plot(result_train_2, linewidth=0.5, label ="Prediction training set 2")
axes[0].set_xlabel("Settlement Period Training Set 2")
axes[0].set_ylabel("Electricity Load [MW]")
#y_values_dates = create_dates(X_future_features[-48*7:].to_numpy(), y_scaler.inverse_transform(y_test[:48*7]))
axes[0].plot(y_train_2,linewidth=0.5, label="Actual")
axes[0].legend()

axes[1].plot(abs(error_train_2), linewidth=0.5, label ="Error hybrid")
axes[1].set_xlabel("Settlement Period Test Set")
axes[1].set_ylabel("Electricity Load [MW]")
axes[1].legend()

########################################################################################################################
# Predictions on the whole training set 2.
########################################################################################################################

fig1, axes1 = plt.subplots(6)
left = 0.1  # the left side of the subplots of the figure
right = 0.9   # the right side of the subplots of the figure
bottom = 0.1  # the bottom of the subplots of the figure
top = 0.9     # the top of the subplots of the figure
wspace = 0.5  # the amount of width reserved for space between subplots,
              # expressed as a fraction of the average axis width
hspace = 0.65  # the amount of height reserved for space between subplots,
              # expressed as a fraction of the average axis height
fig1.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

axes1[0].title.set_text("Predictions on the whole training set (1).")
axes1[0].plot(ANN_train, color = 'blue', linewidth=0.5, label = "Prediction")
axes1[0].plot(y_train_2, color = 'orange', linewidth=0.5, label = "True values")
axes1[0].set_ylabel('(ANN) [MW]')
fig1.legend()

axes1[1].plot(LSTM_train, color = 'blue', linewidth=0.5)
axes1[1].plot(y_train_2, color = 'orange', linewidth=0.5)
axes1[1].set_ylabel('(LSTM) [MW]')

axes1[2].plot(SVR_train, color = 'blue', linewidth=0.5)
axes1[2].plot(y_train_2, color = 'orange', linewidth=0.5)
axes1[2].set_ylabel('(SVR) [MW]')

axes1[3].plot(DT_train, color = 'blue', linewidth=0.5)
axes1[3].plot(y_train_2, color = 'orange', linewidth=0.5)
axes1[3].set_ylabel('(DT) [MW]')

axes1[4].plot(RF_train, color = 'blue', linewidth=0.5)
axes1[4].plot(y_train_2, color = 'orange', linewidth=0.5)
axes1[4].set_ylabel('(RF) [MW]')

axes1[5].plot(result_train_2, color = 'blue', linewidth=0.5)
axes1[5].plot(y_train_2, color = 'orange', linewidth=0.5)
axes1[5].set_ylabel('Hybrid [MW]')

########################################################################################################################
# Predictions on the first 7 days in the training set.
########################################################################################################################

fig2, axes2 = plt.subplots(6)
left = 0.1  # the left side of the subplots of the figure
right = 0.9   # the right side of the subplots of the figure
bottom = 0.1  # the bottom of the subplots of the figure
top = 0.9     # the top of the subplots of the figure
wspace = 0.5  # the amount of width reserved for space between subplots,
              # expressed as a fraction of the average axis width
hspace = 0.65  # the amount of height reserved for space between subplots,
              # expressed as a fraction of the average axis height
fig2.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

axes2[0].title.set_text("Predictions on the first 7 days in the training set (2).")
axes2[0].plot(ANN_train[:48*7], color = 'blue', linewidth=0.5,label = "Prediction")
axes2[0].plot(y_train_2[:48*7], color = 'orange', linewidth=0.5, label = "True values")
axes2[0].set_ylabel('(ANN) [MW]')
fig2.legend()

axes2[1].plot(LSTM_train[:48*7], color = 'blue', linewidth=0.5)
axes2[1].plot(y_train_2[:48*7], color = 'orange', linewidth=0.5)
axes2[1].set_ylabel('(LSTM) [MW]')

axes2[2].plot(SVR_train[:48*7], color = 'blue', linewidth=0.5)
axes2[2].plot(y_train_2[:48*7], color = 'orange', linewidth=0.5)
axes2[2].set_ylabel('(SVR) [MW]')

axes2[3].plot(DT_train[:48*7], color = 'blue', linewidth=0.5)
axes2[3].plot(y_train_2[:48*7], color = 'orange', linewidth=0.5)
axes2[3].set_ylabel('(DT) [MW]')

axes2[4].plot(RF_train[:48*7], color = 'blue', linewidth=0.5)
axes2[4].plot(y_train_2[:48*7], color = 'orange', linewidth=0.5)
axes2[4].set_ylabel('(RF) [MW]')

axes2[5].plot(result_train_2[:48*7], color = 'blue', linewidth=0.5)
axes2[5].plot(y_train_2[:48*7], color = 'orange', linewidth=0.5)
axes2[5].set_ylabel('(Hybrid) [MW]')

########################################################################################################################
# Plot predictions on the whole test set.
########################################################################################################################

fig3, axes3 = plt.subplots(6)
left = 0.1  # the left side of the subplots of the figure
right = 0.9   # the right side of the subplots of the figure
bottom = 0.1  # the bottom of the subplots of the figure
top = 0.9     # the top of the subplots of the figure
wspace = 0.5  # the amount of width reserved for space between subplots,
              # expressed as a fraction of the average axis width
hspace = 0.65  # the amount of height reserved for space between subplots,
              # expressed as a fraction of the average axis height
fig3.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

axes3[0].title.set_text("Predictions on the whole test set.")
axes3[0].plot(ANN_test, color = 'blue', linewidth=0.5,label = "Prediction")
axes3[0].plot(y_test, color = 'orange', linewidth=0.5, label = "True values")
axes3[0].set_ylabel('(ANN) [MW]')
fig3.legend()

axes3[1].plot(LSTM_test, color = 'blue', linewidth=0.5)
axes3[1].plot(y_test, color = 'orange', linewidth=0.5)
axes3[1].set_ylabel('(LSTM) [MW]')

axes3[2].plot(SVR_test, color = 'blue', linewidth=0.5)
axes3[2].plot(y_test, color = 'orange', linewidth=0.5)
axes3[2].set_ylabel('(SVR) [MW]')

axes3[3].plot(DT_test, color = 'blue', linewidth=0.5)
axes3[3].plot(y_test, color = 'orange', linewidth=0.5)
axes3[3].set_ylabel('(DT) [MW]')

axes3[4].plot(RF_test, color = 'blue', linewidth=0.5)
axes3[4].plot(y_test, color = 'orange', linewidth=0.5)
axes3[4].set_ylabel('(RF) [MW]')

axes3[5].plot(result_test, color = 'blue', linewidth=0.5)
axes3[5].plot(y_test, color = 'orange', linewidth=0.5)
axes3[5].set_ylabel('(Hybrid) [MW]')

########################################################################################################################
# Plot predictions on the first 7 days of the test set.
########################################################################################################################

fig4, axes4 = plt.subplots(6)
left = 0.1  # the left side of the subplots of the figure
right = 0.9   # the right side of the subplots of the figure
bottom = 0.1  # the bottom of the subplots of the figure
top = 0.9     # the top of the subplots of the figure
wspace = 0.5  # the amount of width reserved for space between subplots,
              # expressed as a fraction of the average axis width
hspace = 0.65  # the amount of height reserved for space between subplots,
              # expressed as a fraction of the average axis height
fig4.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

axes4[0].title.set_text("Predictions on the first 7 days of the test set.")
axes4[0].plot(ANN_test[:48*7], color = 'blue', linewidth=0.5, label = "Prediction")
axes4[0].plot(y_test[:48*7], color = 'orange', linewidth=0.5, label = "True values")
axes4[0].set_ylabel('(ANN) [MW]')
fig4.legend()

axes4[1].plot(LSTM_test[:48*7], color = 'blue', linewidth=0.5)
axes4[1].plot(y_test[:48*7], color = 'orange', linewidth=0.5)
axes4[1].set_ylabel('[MW]')

axes4[2].plot(SVR_test[:48*7], color = 'blue', linewidth=0.5)
axes4[2].plot(y_test[:48*7], color = 'orange', linewidth=0.5)
axes4[2].set_ylabel('(SVR) [MW]')

axes4[3].plot(DT_test[:48*7], color = 'blue', linewidth=0.5)
axes4[3].plot(y_test[:48*7], color = 'orange', linewidth=0.5)
axes4[3].set_ylabel('(DT) [MW]')

axes4[4].plot(RF_test[:48*7], color = 'blue', linewidth=0.5)
axes4[4].plot(y_test[:48*7], color = 'orange', linewidth=0.5)
axes4[4].set_ylabel('(RF) [MW]')

axes4[5].plot(result_test[:48*7], color = 'blue', linewidth=0.5)
axes4[5].plot(y_test[:48*7], color = 'orange', linewidth=0.5)
axes4[5].set_ylabel('(Hybrid) [MW]')
fig4.show()

########################################################################################################################
# Save the results on the prediction on the test set.
########################################################################################################################

import csv
with open('Compare_Models/MST2_results/Hybrid_result.csv', 'w', newline='',) as file:
    writer = csv.writer(file)
    writer.writerow(["Method","MSE","MAE","RMSE"])
    writer.writerow(["Hybrid",
                     str(mean_squared_error(y_test,result_test)),
                     str(mean_absolute_error(y_test,result_test)),
                     str(np.sqrt(mean_squared_error(y_test,result_test)))
                     ])