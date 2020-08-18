from matplotlib import pylab as plt
import numpy as np
from sklearn.model_selection import train_test_split
from numpy import genfromtxt
from TF_Probability.NN_Random_Weights.Functions import build_model
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.ticker as plticker
import keras
from sklearn.model_selection import train_test_split, TimeSeriesSplit

########################################################################################################################
# Get data and data preprocessing.
########################################################################################################################

# Get the X (containing the features) and y (containing the labels) values
X = pd.read_csv('Data_Preprocessing/For_336_SP_Step_Prediction/X.csv', delimiter=',')
DoW = X["Day of Week"]
X = X.set_index("Time")
dates = X.iloc[:,-1]
X = X.iloc[:,:-6]

y = pd.read_csv('Data_Preprocessing/For_336_SP_Step_Prediction/y.csv', delimiter=',')
y = y.set_index("Time")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, shuffle = False)

X_train = X_train[int(len(X_train)*1/2):]
y_train = y_train[int(len(y_train)*1/2):]
X_test = X_test[:int(len(X_test)*1/2)]
y_test = y_test[:int(len(y_test)*1/2)]
dates = dates[-len(X_train)-len(X_test)*2:-len(X_test)]

# Feature Scaling
x_scaler = StandardScaler()
y_scaler = StandardScaler()
X_train = x_scaler.fit_transform(X_train)
X_test = x_scaler.transform(X_test)
y_train = y_scaler.fit_transform(y_train)

########################################################################################################################
# Create the model.
########################################################################################################################

epochs = 800
learning_rate = 0.001
batches = 19
# Build the model.
model = build_model(X_train.shape[1],learning_rate)

# Extract the loss per epoch to plot the learning progress.
hist_list = pd.DataFrame()

tscv = TimeSeriesSplit()
for train_index, test_index in tscv.split(X_train):
     X_train_split, X_test_split = X_train[train_index], X_train[test_index]
     y_train_split, y_test_split = y_train[train_index], y_train[test_index]
     hist_split = model.fit(X_train_split, y_train_split, epochs = epochs, batch_size = batches, verbose = 2)

# Describe model.
model.summary()

# Plot the learning progress.
fig1, axs1=plt.subplots(1,1,figsize=(4,4))
axs1.plot(model.history.history["mean_absolute_error"], color = "blue")
axs1.set_ylabel('Loss')
axs1.set_xlabel('Epochs')
axs1.grid(True)
fig1.show()
fig1.savefig("TF_Probability/NN_Random_Weights/Figures/Loss_Epochs.pdf", bbox_inches='tight')

# Save or load the model
model.save("TF_Probability/NN_Random_Weights/DMST_NN_w_Rd_Weights.h5")
#model = keras.models.load_model("TF_Probability/NN_Random_Weights/DMST_NN_w_Rd_Weights.h5")

########################################################################################################################
# Predicting the generation.
########################################################################################################################

# Make 350 predictions on the test set and calculate the errors for each prediction.
yhats_test = [model.predict(X_test) for _ in range(350)]
predictions_test = np.array(yhats_test)
predictions_test = predictions_test.reshape(-1,len(predictions_test[1]))

yhats_train = [model.predict(X_train) for _ in range(350)]
predictions_train = np.array(yhats_train)
predictions_train = predictions_train.reshape(-1,len(predictions_train[1]))

predictions_test = y_scaler.inverse_transform(predictions_test)/1000
predictions_train = y_scaler.inverse_transform(predictions_train)/1000

X_train = x_scaler.inverse_transform(X_train)
X_test = x_scaler.inverse_transform(X_test)
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
          label = "Training Set", color = "blue")
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
axs2[0].set_ylabel('Load, GW', size = 14)
axs2[0].plot(30, 30, color = "red", label = 'Error')

axs2[1].plot(dates.iloc[-len(X_test)-48*7*2:-len(X_test)-48*7+1],
             error_train_plot,
             label = "Error", color = "red")
axs2[1].set_xlabel('Date (2019)',size = 14)
axs2[1].set_ylabel('Error, GW',size = 14)

# Include additional details such as tick intervals
loc = plticker.MultipleLocator(base=48) # Puts ticks at regular intervals
axs2[0].xaxis.set_major_locator(loc), axs2[1].xaxis.set_major_locator(loc)
axs2[0].grid(True), axs2[1].grid(True)
fig2.autofmt_xdate(rotation = 0)
axs2[0].legend(loc = (1.02, 0.6))
plt.xticks(np.arange(1,339, 48), ["14:00\n07/11","14:00\n07/12","14:00\n07/13",
                                  "14:00\n07/14","14:00\n07/15","14:00\n07/16",
                                  "14:00\n07/17","14:00\n07/18"])
fig2.show()
fig2.savefig("TF_Probability/NN_Random_Weights/Figures/DMST_Train_Set_Pred.pdf", bbox_inches='tight')

########################################################################################################################
# Calculate the stddev from the 350 predictions.
########################################################################################################################

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
          label = "Training Set", color = "blue")
axs3[0].plot(dates.iloc[-len(X_test):-len(X_test)+48*7],
          mean_test[:48*7],
          label = "Mean of the predictions\nNN with random weights", color = "orange")
axs3[0].fill_between(dates.iloc[-len(X_test):-len(X_test)+48*7+1],
                     (mean_test+stddev_test)[:48*7+1].reshape(-1,),
                     (mean_test-stddev_test)[:48*7+1].reshape(-1,),
                     label = "+- 1x Standard Deviation",
                     alpha=0.2, color = "orange")
axs3[0].plot(dates.iloc[-len(X_test):-len(X_test)+48*7+1],
          y_test[:48*7+1],
          label = "Test Set", color = "black")
axs3[0].set_ylabel('Load, GW', size = 14)
axs3[0].axvline(dates.iloc[-len(X_test)], linestyle="--", color = "black")
axs3[0].plot(30, 30, color = "red", label = "Error")

axs3[1].plot(dates.iloc[-len(X_test)-48*3:-len(X_test)+48*7+1],
             error_test_plot,
             label = "Error", color = "red")
axs3[1].axvline(dates.iloc[-len(X_test)], linestyle="--", color = "black")
axs3[1].set_xlabel('Date (2019)',size = 14)
axs3[1].set_ylabel('Error, GW',size = 14)

# Include additional details such as tick intervals
loc = plticker.MultipleLocator(base=48) # Puts ticks at regular intervals
axs3[0].xaxis.set_major_locator(loc), axs3[1].xaxis.set_major_locator(loc)
axs3[0].grid(True), axs3[1].grid(True)
fig3.autofmt_xdate(rotation = 0)
axs3[0].legend(loc = (1.02, 0.48))
plt.xticks(np.arange(1,482, 48), ["14:00\n07/22","14:00\n07/23","14:00\n07/24",
                                  "14:00\n07/25","14:00\n07/26","14:00\n07/27",
                                  "14:00\n07/28","14:00\n07/29","14:00\n07/30",
                                  "14:00\n07/31","14:00\n08/01"])
fig3.show()
fig3.savefig("TF_Probability/NN_Random_Weights/Figures/DMST_Test_Set_Pred.pdf", bbox_inches='tight')

########################################################################################################################
# Calculate the errors on the training and the test set.
########################################################################################################################

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

# Plot the histograms of the 2 sets.
fig4, axs4=plt.subplots(1,2,figsize=(12,6))
axs4[0].grid(True)
axs4[0].hist(error_column_train, bins = 50, color = "blue")
axs4[0].set_xlabel("Prediction Error on Training Set [GW]", size = 14)
axs4[0].set_ylabel("Count", size = 14)

axs4[1].grid(True)
axs4[1].hist(error_column_test, bins = 50, color = "blue")
axs4[1].set_xlabel("Prediction Error on Test Set [GW]", size = 14)
axs4[1].set_ylabel("Count", size = 14)
fig4.show()
fig4.savefig("TF_Probability/NN_Random_Weights/Figures/DMST_Histograms_Train_and_Test_Set_Error_Pred.pdf", bbox_inches='tight')

########################################################################################################################
# Plot the mean and standard deviation per week for the training and test set.
########################################################################################################################

X = pd.read_csv('Data_Preprocessing/For_336_SP_Step_Prediction/X.csv', delimiter=',')
sp_train = X["Settlement Period"][-len(X_test)*2-len(X_train):-len(X_test)*2].values+(48*DoW[-len(X_test)*2-len(X_train):-len(X_test)*2]).values
sp_train_column = np.array([sp_train]*350).reshape(-1,)
# Create a dataframe that contains the SPs (1-336) and the load values.
error_train = pd.DataFrame({'SP':sp_train_column, 'Error_Train': error_column_train[:len(sp_train_column)]})

# Compute the mean and variation for each x.
training_stats = pd.DataFrame({'Index':np.linspace(1,336,336),
                               'Mean':np.linspace(1,336,336),
                               'Stddev':np.linspace(1,336,336)})

for i in range(1,337):
    training_stats.iloc[i-1,1]=np.mean(error_train[error_train["SP"]==i].iloc[:,-1])
    training_stats.iloc[i-1,2]=np.std(error_train[error_train["SP"]==i].iloc[:,-1])

settlement_period_test = X["Settlement Period"][-len(X_test)*2:-len(X_test)].values+(48*DoW[-len(X_test)*2:-len(X_test)]).values
sp_test_column = np.array([settlement_period_test]*350).reshape(-1,)
# Create a dataframe that contains the SPs (1-336) and the load values.
error_test = pd.DataFrame({'SP':sp_test_column, 'Error_Test': error_column_test[:len(sp_test_column)]})

# Plot the projected errors onto a single week to see the variation in the timeseries.
test_stats = pd.DataFrame({'Index':np.linspace(1,336,336),
                               'Mean':np.linspace(1,336,336),
                               'Stddev':np.linspace(1,336,336)})

for i in range(1,337):
    test_stats.iloc[i-1,1]=np.mean(error_test[error_test["SP"]==i].iloc[:,-1])
    test_stats.iloc[i-1,2]=np.std(error_test[error_test["SP"]==i].iloc[:,-1])

# Plot the projected errors onto a single week to see the variation in the timeseries.
fig5, axs5=plt.subplots(2,1,figsize=(12,10))

# Plot the mean and standard deviation of the errors that are made on the training set.
axs5[0].plot(training_stats.iloc[:,0],
          training_stats.iloc[:,1],
          color = "orange", label = "Mean of all projected errors (Training Set)")
axs5[0].fill_between(training_stats.iloc[:,0],
                  (training_stats.iloc[:,1]-training_stats.iloc[:,2]),
                  (training_stats.iloc[:,1]+training_stats.iloc[:,2]),
                  alpha=0.2, color = "orange", label = "+- 1x Standard Deviation")
axs5[0].set_ylabel("Error when training\nNN w. random weights, GW", size = 14)

axs5[1].plot(test_stats.iloc[:,0],
          test_stats.iloc[:,1],
          color = "orange", label = "Mean of all projected errors (Test Set)")
axs5[1].fill_between(test_stats.iloc[:,0],
                  (test_stats.iloc[:,1]-test_stats.iloc[:,2]),
                  (test_stats.iloc[:,1]+test_stats.iloc[:,2]),
                  alpha=0.2, color = "orange", label = "+- 1x Standard Deviation")
axs5[1].set_ylabel("Error when testing\nNN w. random weights, GW", size = 14)

# Include additional details such as tick intervals, legend positioning and grid on.
axs5[0].minorticks_on(), axs5[1].minorticks_on()
axs5[0].grid(b=True, which='major'), axs5[0].grid(b=True, which='minor',alpha = 0.2)
axs5[1].grid(b=True, which='major'), axs5[1].grid(b=True, which='minor',alpha = 0.2)
axs5[0].set_xticks(np.arange(1,385, 24)), axs5[1].set_xticks(np.arange(1,385, 24))
axs5[0].set_xticklabels(["00:00\nMonday","12:00",
                       "00:00\nTuesday","12:00",
                       "00:00\nWednesday", "12:00",
                       "00:00\nThursday", "12:00",
                       "00:00\nFriday","12:00",
                       "00:00\nSaturday", "12:00",
                       "00:00\nSunday","12:00",
                       "00:00"])
axs5[1].set_xticklabels(["00:00\nMonday","12:00",
                       "00:00\nTuesday","12:00",
                       "00:00\nWednesday", "12:00",
                       "00:00\nThursday", "12:00",
                       "00:00\nFriday","12:00",
                       "00:00\nSaturday", "12:00",
                       "00:00\nSunday","12:00",
                       "00:00"])

axs5[0].legend(fontsize=12), axs5[1].legend(fontsize=12)
axs5[0].tick_params(axis = "both", labelsize = 12), axs5[1].tick_params(axis = "both", labelsize = 12)
fig5.show()
fig5.savefig("TF_Probability/NN_Random_Weights/Figures/DMST_Mean_and_Stddev_of_Error_Train_and_Test_Set_Pred.pdf", bbox_inches='tight')

########################################################################################################################
# Plot the standard deviation per week for the test and training set.
########################################################################################################################

zeros = np.zeros((336,))

fig7, axs7=plt.subplots(2,1,figsize=(12,10))
axs7[0].fill_between(training_stats.iloc[:,0],
                  zeros,
                  +training_stats.iloc[:,2],
                  alpha=0.2, color = "orange", label = "Error when training the NN w. random weigths")
axs7[0].set_ylabel("Standard deviation, electricity load, GW", size = 14)

axs7[1].fill_between(test_stats.iloc[:,0],
                  zeros,
                  test_stats.iloc[:,2],
                  alpha=0.2, color = "orange", label = "Error when testing the NN w. random weigths")
axs7[1].set_ylabel("Standard deviation, electricity load, GW", size = 14)

# Include additional details such as tick intervals, legend positioning and grid on.
axs7[0].minorticks_on(), axs7[1].minorticks_on()
axs7[0].grid(b=True, which='major'), axs7[0].grid(b=True, which='minor',alpha = 0.2)
axs7[1].grid(b=True, which='major'), axs7[1].grid(b=True, which='minor',alpha = 0.2)
axs7[0].set_xticks(np.arange(1,385, 24)), axs7[1].set_xticks(np.arange(1,385, 24))
axs7[1].set_xticklabels(["00:00\nMonday","12:00",
                       "00:00\nTuesday","12:00",
                       "00:00\nWednesday", "12:00",
                       "00:00\nThursday", "12:00",
                       "00:00\nFriday","12:00",
                       "00:00\nSaturday", "12:00",
                       "00:00\nSunday","12:00",
                       "00:00"])
axs7[0].set_xticklabels(["00:00\nMonday","12:00",
                       "00:00\nTuesday","12:00",
                       "00:00\nWednesday", "12:00",
                       "00:00\nThursday", "12:00",
                       "00:00\nFriday","12:00",
                       "00:00\nSaturday", "12:00",
                       "00:00\nSunday","12:00",
                       "00:00"])
axs7[0].legend(fontsize=14), axs7[1].legend(fontsize=14)
axs7[0].tick_params(axis = "both", labelsize = 12), axs7[1].tick_params(axis = "both", labelsize = 12)
axs7[0].set_ylim([0,5.4]), axs7[1].set_ylim([0,5.4])

fig7.show()
fig7.savefig("TF_Probability/NN_Random_Weights/Figures/Stddev_of_Error_Test_and_Traing_Set_Pred.pdf", bbox_inches='tight')

########################################################################################################################
# Save the results in a csv file.
########################################################################################################################

# Calculate the errors from the mean to the actual vaules.
print("-"*200)
print("The mean absolute error of the training set is %0.2f GW." % np.mean(abs(error_train.iloc[:,-1])))
print("The mean squared error of the training set is %0.2f GW." % np.mean(error_train.iloc[:,-1]**2))
print("The root mean squared error of the training set is %0.2f GW." % np.sqrt(np.mean(error_train.iloc[:,-1]**2)))
print("-"*200)
print("The mean absolute error of the test set is %0.2f GW." % np.mean(abs(error_test.iloc[:,-1])))
print("The mean squared error of the test set is %0.2f GW." % np.mean(error_test.iloc[:,-1]**2))
print("The root mean squared error of the test set is %0.2f GW." % np.sqrt(np.mean(error_test.iloc[:,-1]**2)))
print("-"*200)

df_errors = pd.DataFrame({"MSE_Train": [np.mean(error_train.iloc[:,-1]**2)],
                          "MAE_Train": [np.mean(abs(error_train.iloc[:,-1]))],
                          "RMSE_Train": [np.sqrt(np.mean(error_train.iloc[:,-1]**2))],
                          "MSE_Test": [np.mean(error_test.iloc[:,-1]**2)],
                          "MAE_Test": [np.mean(abs(error_test.iloc[:,-1]))],
                          "RMSE_Test": [np.sqrt(np.mean(error_test.iloc[:,-1]**2))],
                          })
df_errors.to_csv("Compare_Models/Direct_Multi_Step_Probability_Results/Probability_Based_on_Model/NN_error.csv")

training_stats.to_csv("Compare_Models/Direct_Multi_Step_Probability_Results/Probability_Based_on_Model/NN_mean_errors_stddevs_train.csv")
test_stats.to_csv("Compare_Models/Direct_Multi_Step_Probability_Results/Probability_Based_on_Model/NN_mean_errors_stddevs_test.csv")

