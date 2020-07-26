from matplotlib import pylab as plt
import numpy as np
from sklearn.model_selection import train_test_split
from numpy import genfromtxt
from TF_Probability.MST_Rd_Weights.Functions import build_model
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.ticker as plticker

########################################################################################################################
# Get data and data preprocessing.
########################################################################################################################

# Get the X (containing the features) and y (containing the labels) values
X = pd.read_csv('Data_Preprocessing/For_Multi_Step_Prediction/X.csv', delimiter=',')
DoW = X["Day of Week"]
X = X.set_index("Time")
dates = X.iloc[:,-1]
X = X.iloc[:,:-5]

y = pd.read_csv('Data_Preprocessing/For_Multi_Step_Prediction/y.csv', delimiter=',')
y = y.set_index("Time")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = False)

X_train = X_train[int(len(X_train)*1/2):]
y_train = y_train[int(len(y_train)*1/2):]
dates = dates[-len(X_train)-len(X_test):]

# Feature Scaling
x_scaler = StandardScaler()
y_scaler = StandardScaler()
X_train = x_scaler.fit_transform(X_train)
X_test = x_scaler.transform(X_test)
y_train = y_scaler.fit_transform(y_train)

########################################################################################################################
# Create the model.
########################################################################################################################

epochs =8000
learning_rate = 0.001
batches = 32

# Build the model.
model = build_model(X_train.shape[1],learning_rate)
# Run training session.
model.fit(X_train,y_train, epochs=epochs, batch_size=batches, verbose=2)
# Describe model.
model.summary()

# Plot the learning progress.
fig1, axs1=plt.subplots(1,1,figsize=(4,6))
axs1.plot(model.history.history["mean_absolute_error"], color = "blue")
axs1.set_ylabel('Loss')
axs1.set_xlabel('Epochs')
fig1.show()

# Save or load the model
#model.save("TF_Probability/MST_Rd_Weights/SMST_No_Date.h5")
#model = keras.models.load_model("Electricity_Generation_Prediction/ANN/Single_Multi_Step_Prediction/SMST_No_Date.h5")

########################################################################################################################
# Predicting the generation.
########################################################################################################################

# Make a 350 predictions on the test set and calculate the errors for each prediction.
yhats_test = [model.predict(X_test) for _ in range(350)]
predictions_test = np.array(yhats_test)
predictions_test = predictions_test.reshape(-1,len(predictions_test[1]))

yhats_train = [model.predict(X_train) for _ in range(350)]
predictions_train = np.array(yhats_train)
predictions_train = predictions_train.reshape(-1,len(predictions_train[1]))

predictions_test = y_scaler.inverse_transform(predictions_test)/1000
predictions_train = y_scaler.inverse_transform(predictions_train)/1000

X_train = x_scaler.inverse_transform(X_train)
X_train[:,0] = X_train[:,0]/1000
X_test = x_scaler.inverse_transform(X_test)
X_test[:,0] = X_test[:,0]/1000
y_train = (y_scaler.inverse_transform(y_train)/1000).reshape(-1,)
y_test = np.array(y_test.iloc[:,-1]/1000).reshape(-1,)

########################################################################################################################
# Data processing for plotting curves and printing the errors.
########################################################################################################################

# Calculate the stddev from the 350 predictions.
mean_train = (sum(predictions_train)/350).reshape(-1,1)
stddev_train = np.zeros((len(mean_train),1))
for i in range(len(X_train)):
    stddev_train[i,0] = np.std(predictions_train[:,i])

error_train = mean_train.reshape(-1,) - y_train
error_train_plot = error_train[-48*7*2:-48*7+1].reshape(-1,1)

# Plot the result with the truth in blue and the predictions in orange.
fig2, axs2=plt.subplots(2,1,figsize=(12,6))
axs2[0].plot(dates.iloc[-len(X_test)-48*7*2:-len(X_test)-48*7+1],
          y_train[-48*7*2:-48*7+1],
          label = "Training Set (True Values)", color = "blue")
axs2[0].plot(dates.iloc[-len(X_test)-48*7*2:-len(X_test)-48*7+1],
          mean_train[-48*7*2:-48*7+1],
          label = "Mean of the predictions", color = "orange")

# Potentially include all the predictions made
#fig2.plot(X_axis[-48*7:], predictions.T[:,:50], alpha = 0.1, color = "orange")

axs2[0].fill_between(dates.iloc[-len(X_test)-48*7*2:-len(X_test)-48*7+1],
                  (mean_train + stddev_train)[-48*7*2:-48*7+1].reshape(-1,),
                  (mean_train - stddev_train)[-48*7*2:-48*7+1].reshape(-1,),
                  label = "+- 1x Standard Deviation",
                  alpha=0.2, color = "orange")
axs2[0].set_ylabel('Load [GW]', size = 14)

axs2[1].plot(dates.iloc[-len(X_test)-48*7*2:-len(X_test)-48*7+1],
             error_train_plot,
             label = "Absolute Error", color = "red")
axs2[1].set_xlabel('Date',size = 14)
axs2[1].set_ylabel('Absolute Error [GW]',size = 14)

# Include additional details such as tick intervals
loc = plticker.MultipleLocator(base=48) # Puts ticks at regular intervals
axs2[0].xaxis.set_major_locator(loc)
axs2[1].xaxis.set_major_locator(loc)
axs2[0].grid(True)
axs2[1].grid(True)
fig2.autofmt_xdate(rotation = 12)
axs2[0].legend(loc = (1.04, 0.7))
axs2[1].legend(loc = (1.04, 0.9))
fig2.show()

# Calculate the stddev from the 350 predictions.
mean_test = (sum(predictions_test)/350).reshape(-1,1)
stddev_test = np.zeros((len(mean_test),1))
for i in range (len(X_test)):
    stddev_test[i,0] = np.std(predictions_test[:,i])

error_test = mean_test.reshape(-1,) - y_test
error_test_plot = np.zeros((48*3+48*7+1,1))
error_test_plot[-336:] = error_test[:48*7].reshape(-1,1)

# Plot the result with the truth in blue and the predictions in orange.
fig3, axs3=plt.subplots(2,1,figsize=(12,6))
axs3[0].plot(dates.iloc[-len(X_test)-48*3:-len(X_test)],
          y_train[-48*3:],
          label = "Training Set (True Values)", color = "blue")
axs3[0].plot(dates.iloc[-len(X_test):-len(X_test)+48*7],
          mean_test[:48*7],
          label = "Mean of the predictions", color = "orange")

# Potentially include all the predictions made
#fig3.plot(X_axis[-48*7:], predictions.T[:,:50], alpha = 0.1, color = "orange")

axs3[0].fill_between(dates.iloc[-len(X_test):-len(X_test)+48*7+1],
                     (mean_test+stddev_test)[:48*7+1].reshape(-1,),
                     (mean_test-stddev_test)[:48*7+1].reshape(-1,),
                     label = "+- 1x Standard Deviation",
                     alpha=0.2, color = "orange")
axs3[0].plot(dates.iloc[-len(X_test):-len(X_test)+48*7+1],
          y_test[:48*7+1],
          label = "Test Set (True Values)", color = "black")
axs3[0].set_ylabel('Load [GW]', size = 14)
axs3[0].axvline(dates.iloc[-len(X_test)], linestyle="--", color = "black")

axs3[1].plot(dates.iloc[-len(X_test)-48*3:-len(X_test)+48*7+1],
             error_test_plot,
             label = "Absolute Error", color = "red")
axs3[1].axvline(dates.iloc[-len(X_test)], linestyle="--", color = "black")
axs3[1].set_xlabel('Date',size = 14)
axs3[1].set_ylabel('Absolute Error [GW]',size = 14)

# Include additional details such as tick intervals
loc = plticker.MultipleLocator(base=48) # Puts ticks at regular intervals
axs3[0].xaxis.set_major_locator(loc)
axs3[1].xaxis.set_major_locator(loc)
axs3[0].grid(True)
axs3[1].grid(True)
fig3.autofmt_xdate(rotation = 12)
axs3[0].legend(loc = (1.04, 0.6))
axs3[1].legend(loc = (1.04, 0.9))
fig3.show()

# Make one large column vector containing all the predictions
pred_train_vector = predictions_train.reshape(-1, 1)
pred_test_vector = predictions_test.reshape(-1, 1)

# Create a np array with all the predictions of the test set in the first column and the corresponding error in the next column.
predictions_and_errors_test = np.zeros((len(pred_test_vector),2))
predictions_and_errors_test[:,:-1] = pred_test_vector
j=0
for i in range (len(pred_test_vector)):
        predictions_and_errors_test[i,1] = pred_test_vector[i]-y_test[j]
        j=j+1
        if j == len(y_test):
            j=0

# Create a np array with all the predictions of the trianing set in the first column and the corresponding error in the next column.
predictions_and_errors_train = np.zeros((len(pred_train_vector),2))
predictions_and_errors_train[:,:-1] = pred_train_vector
j=0
for i in range (len(pred_train_vector)):
        predictions_and_errors_train[i,1] = pred_train_vector[i]-y_train[j]
        j=j+1
        if j == len(y_train):
            j=0

# Plot the histogram of the errors.
error_column_test = predictions_and_errors_test[:,1]
error_column_train = predictions_and_errors_train[:,1]
# for i in range(60):
#     error_column = np.delete(error_column,np.argmax(error_column),0)
#     #error_column = np.delete(error_column,np.argmin(error_column),0)

# Plot the histograms of the 2 SPs.
fig3, axs3=plt.subplots(1,2,figsize=(12,6))
axs3[1].hist(error_column_test, bins = 50, color = "blue")
axs3[1].set_xlabel("Prediction Error on Test Set [GW]", size = 14)
axs3[1].set_ylabel("Count", size = 14)

axs3[0].hist(error_column_train, bins = 50, color = "blue")
axs3[0].set_xlabel("Prediction Error on Training Set [GW]", size = 14)
axs3[0].set_ylabel("Count", size = 14)
fig3.show()

# Calculate the errors from the mean to the actual vaules.
print("-"*200)
error = np.abs(y_test-mean_test)
print("The mean absolute error of the test set is %0.2f" % np.mean(error))
print("The mean squared error of the test set is %0.2f" % np.mean(error**2))
print("The root mean squared error of the test set is %0.2f" % np.sqrt(np.mean(error**2)))
print("-"*200)

stats = np.concatenate((mean_test, stddev_test, y_test.reshape(-1,1)), axis = 1)

########################################################################################################################
# Save the results in a csv file.
########################################################################################################################

import csv
with open('TF_Probability/Results/NN_mean_error.csv', 'w', newline='', ) as file:
    writer = csv.writer(file)
    writer.writerow(["Method","MSE","MAE","RMSE"])
    writer.writerow(["NN",
                     str(np.mean(error**2)),
                     str(np.mean(error)),
                     str(np.sqrt(np.mean(error**2)))
                     ])

stats = pd.DataFrame(stats)
stats.columns = ["Mean", "Stddev", "Test_Set"]
stats.to_csv("TF_Probability/Results/NN_prediction.csv", index = False)

########################################################################################################################
# Plot the mean and standard deviation per week for the training set.
########################################################################################################################

settlement_period_train = X["Settlement Period"][-len(X_test)-len(X_train):-len(X_test)].values+(48*DoW[-len(X_test)-len(X_train):-len(X_test)]).values
long_column = np.array([settlement_period_train]*350).reshape(-1,)
# Create a dataframe that contains the SPs (1-336) and the load values.
error_train = pd.DataFrame({'SP':long_column, 'Error_Train': error_column_train[:len(long_column)]})

# Plot the projected errors onto a single week to see the variation in the timeseries.
fig4, axs4=plt.subplots(2,1,figsize=(12,10))
axs4[0].scatter(error_train["SP"],
             error_train["Error_Train"],
             alpha=0.05, label = "Projected Errors (Training Set)", color = "red")
axs4[0].set_ylabel("Error during training [GW]", size = 14)
axs4[0].set_xticks(np.arange(1,385, 48))
axs4[0].set_xticklabels(["1 / Monday", "49 / Tuesday", "97 / Wednesday", "145 / Thursday", "193 / Friday","241 / Saturday", "289 / Sunday",""])
axs4[0].grid(True)
axs4[0].legend()

# Compute the mean and variation for each x.
training_stats = pd.DataFrame({'Index':np.linspace(1,336,336),
                               'Mean':np.linspace(1,336,336),
                               'Stddev':np.linspace(1,336,336)})

for i in range(1,337):
    training_stats.iloc[i-1,1]=np.mean(error_train[error_train["SP"]==i].iloc[:,-1])
    training_stats.iloc[i-1,2]=np.std(error_train[error_train["SP"]==i].iloc[:,-1])

# Plot the mean and standard deviation of the errors that are made on the training set.
axs4[1].plot(training_stats.iloc[:,0],
          training_stats.iloc[:,1],
          color = "orange", label = "Mean of all projected errors  (Training Set)")
axs4[1].fill_between(training_stats.iloc[:,0],
                  (training_stats.iloc[:,1]-training_stats.iloc[:,2]),
                  (training_stats.iloc[:,1]+training_stats.iloc[:,2]),
                  alpha=0.2, color = "orange", label = "+- 1x Standard Deviation")
axs4[1].set_ylabel("Error during training [GW]", size = 14)
axs4[1].set_xlabel("Settlement Period / Weekday", size = 14)
axs4[1].set_xticks(np.arange(1,385, 48))
axs4[1].set_xticklabels(["1 / Monday", "49 / Tuesday", "97 / Wednesday", "145 / Thursday", "193 / Friday","241 / Saturday", "289 / Sunday",""])
axs4[1].legend()
axs4[1].grid(True)
fig4.show()

########################################################################################################################
# Plot the mean and standard deviation per week for the test set.
########################################################################################################################

settlement_period_test = X["Settlement Period"][-len(X_test):].values+(48*DoW[-len(X_test):]).values
long_column = np.array([settlement_period_test]*350).reshape(-1,)
# Create a dataframe that contains the SPs (1-336) and the load values.
error_test = pd.DataFrame({'SP':long_column, 'Error_Test': error_column_test[:len(long_column)]})

# Plot the projected errors onto a single week to see the variation in the timeseries.
fig5, axs5=plt.subplots(2,1,figsize=(12,10))
axs5[0].scatter(error_test["SP"],
             error_test["Error_Test"],
             alpha=0.05, label = "Projected Errors (Test Set)", color = "red")
axs5[0].set_ylabel("Error during test [GW]", size = 14)
axs5[0].set_xticks(np.arange(1,385, 48))
axs5[0].set_xticklabels(["1 / Monday", "49 / Tuesday", "97 / Wednesday", "145 / Thursday", "193 / Friday","241 / Saturday", "289 / Sunday",""])
axs5[0].grid(True)
axs5[0].legend()

# Compute the mean and variation for each x.
test_stats = pd.DataFrame({'Index':np.linspace(1,336,336),
                               'Mean':np.linspace(1,336,336),
                               'Stddev':np.linspace(1,336,336)})

for i in range(1,337):
    test_stats.iloc[i-1,1]=np.mean(error_test[error_test["SP"]==i].iloc[:,-1])
    test_stats.iloc[i-1,2]=np.std(error_test[error_test["SP"]==i].iloc[:,-1])

# Plot the mean and standard deviation of the errors that are made on the test set.
axs5[1].plot(test_stats.iloc[:,0],
          test_stats.iloc[:,1],
          color = "orange", label = "Mean of all projected errors (Test Set)")
axs5[1].fill_between(test_stats.iloc[:,0],
                  (test_stats.iloc[:,1]-test_stats.iloc[:,2]),
                  (test_stats.iloc[:,1]+test_stats.iloc[:,2]),
                  alpha=0.2, color = "orange", label = "+- 1x Standard Deviation")
axs5[1].set_ylabel("Error during test [GW]", size = 14)
axs5[1].set_xlabel("Settlement Period / Weekday", size = 14)
axs5[1].set_xticks(np.arange(1,385, 48))
axs5[1].set_xticklabels(["1 / Monday", "49 / Tuesday", "97 / Wednesday", "145 / Thursday", "193 / Friday","241 / Saturday", "289 / Sunday",""])
axs5[1].legend()
axs5[1].grid(True)
fig5.show()
