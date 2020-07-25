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

epochs = 500
learning_rate = 0.001
batches = 64

# Build the model.
model = build_model(X_train.shape[1],learning_rate)
# Run training session.
model.fit(X_train,y_train, epochs=epochs, batch_size=batches, verbose=2)
# Describe model.
model.summary()

# Plot the learning progress.
plt.plot(model.history.history["mean_absolute_error"][150:], color = "blue")
plt.show()

# Save or load the model
#model.save("TF_Probability/MST_Rd_Weights/SMST_No_Date.h5")
#model = keras.models.load_model("Electricity_Generation_Prediction/ANN/Single_Multi_Step_Prediction/SMST_No_Date.h5")

# Make a 1000 predictions on the test set and calculate the errors for each prediction.
yhats = [model.predict(X_test) for _ in range(500)]
predictions = np.array(yhats)
predictions = predictions.reshape(-1,len(predictions[1]))

########################################################################################################################
# Predicting the generation.
########################################################################################################################

pred_train = y_scaler.inverse_transform(model.predict(X_train))/1000
pred_train = pred_train.reshape(-1,)
pred_test = y_scaler.inverse_transform(model.predict(X_test))/1000
pred_test = pred_test.reshape(-1,)
predictions = y_scaler.inverse_transform(predictions)/1000

X_train = x_scaler.inverse_transform(X_train)
X_train[:,0] = X_train[:,0]/1000
X_test = x_scaler.inverse_transform(X_test)
X_test[:,0] = X_test[:,0]/1000
y_train = (y_scaler.inverse_transform(y_train)/1000).reshape(-1,)
y_test = np.array(y_test.iloc[:,-1]/1000).reshape(-1,)

########################################################################################################################
# Data processing for plotting curves and printing the errors.
########################################################################################################################

# Calculate the stddev from the 500 predictions.
mean = (sum(predictions)/500).reshape(-1,1)
stddev = np.zeros((len(mean),1))
for i in range (48*7):
    stddev[i,0] = np.std(predictions[:,i])

mean_plus_stddev = mean+stddev
mean_minus_stddev = mean-stddev

error_test = mean.reshape(-1,) - y_test
error_test_plot = np.zeros((48*3+48*7,1))
error_test_plot[-336:] = error_test[:48*7].reshape(-1,1)

# Plot the result with the truth in blue and the predictions in orange.
fig2, axs2=plt.subplots(2,1,figsize=(12,6))
axs2[0].plot(dates.iloc[-len(X_test)-48*3:-len(X_test)],
          y_train[-48*3:],
          label = "Training Set (True Values)", color = "blue")
axs2[0].plot(dates.iloc[-len(X_test):-len(X_test)+48*7],
          mean[:48*7],
          label = "Mean of the predictions", color = "orange")

# Potentially include all the predictions made
#fig2.plot(X_axis[-48*7:], predictions.T[:,:50], alpha = 0.1, color = "orange")

axs2[0].fill_between(dates.iloc[-len(X_test):-len(X_test)+48*7],
                  mean_plus_stddev[:48*7].reshape(-1,),
                  mean_minus_stddev[:48*7].reshape(-1,),
                  label = "+- 1x Standard Deviation",
                  alpha=0.3, color = "orange")
axs2[0].plot(dates.iloc[-len(X_test):-len(X_test)+48*7+1],
          y_test[:48*7+1],
          label = "Test Set (True Values)", color = "black")
axs2[0].set_ylabel('Load [GW]')
axs2[0].axvline(dates.iloc[-len(X_test)], linestyle="--", color = "black")

axs2[1].plot(dates.iloc[-len(X_test)-48*3:-len(X_test)+48*7],
             error_test_plot,
             label = "Absolute Error", color = "red")
axs2[1].axvline(dates.iloc[-len(X_test)], linestyle="--", color = "black")
axs2[1].set_xlabel('Date',size = 14)
axs2[1].set_ylabel('Absolute Error [GW]',size = 14)

# Include additional details such as tick intervals
loc = plticker.MultipleLocator(base=48) # Puts ticks at regular intervals
axs2[0].xaxis.set_major_locator(loc)
axs2[1].xaxis.set_major_locator(loc)
axs2[0].grid(True)
axs2[1].grid(True)
fig2.autofmt_xdate(rotation = 12)
axs2[0].legend()
axs2[1].legend()
fig2.show()

# Make one large column vector containing all the predictions
predictions_vector = predictions.reshape(-1, 1)

# Create a np array with all the predictions in the first column and the corresponding error in the next column.
predictions_and_errors = np.zeros((len(predictions_vector),2))
predictions_and_errors[:,:-1] = predictions_vector
j=0
for i in range (len(predictions_vector)):
        predictions_and_errors[i,1] = predictions_vector[i]-y_test[j]
        j=j+1
        if j == len(y_test):
            j=0

# Plot the histogram of the errors.
error_column = predictions_and_errors[:,1]
# for i in range(60):
#     error_column = np.delete(error_column,np.argmax(error_column),0)
#     #error_column = np.delete(error_column,np.argmin(error_column),0)

# Plot the histograms of the 2 SPs.
fig3, axs3=plt.subplots(1,1,figsize=(12,6))
axs3.hist(error_column, bins = 50, color = "blue")
axs3.set_xlabel("Prediction Error [GW]", size = 14)
axs3.set_ylabel("Count", size = 14)
fig3.show()

# Calculate the errors from the mean to the actual vaules.
print("-"*200)
error = np.abs(y_test-pred_test)
print("The mean absolute error of the test set is %0.2f" % np.mean(error))
print("The mean squared error of the test set is %0.2f" % np.mean(error**2))
print("The root mean squared error of the test set is %0.2f" % np.sqrt(np.mean(error**2)))
print("-"*200)

stats = np.concatenate((mean, stddev, y_test.reshape(-1,1)), axis = 1)

########################################################################################################################
# Save the results in a csv file.
########################################################################################################################

import csv
with open('TF_Probability/Results/NN_error.csv', 'w', newline='', ) as file:
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
