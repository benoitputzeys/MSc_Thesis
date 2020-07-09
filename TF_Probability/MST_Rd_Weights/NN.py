from matplotlib import pylab as plt
import numpy as np
from sklearn.model_selection import train_test_split
from numpy import genfromtxt
from TF_Probability.MST_Rd_Weights.Functions import build_model
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.ticker as plticker

# Get the X (containing the features) and y (containing the labels) values
X = pd.read_csv('Data_Entsoe/Data_Preprocessing/For_Multi_Step_Prediction/X_API.csv', delimiter=',')
X = X.drop(['Unnamed: 0'], axis=1)
dates = X.iloc[:,-1]
X = X.iloc[-48*7*5:,:-1]

y = pd.read_csv('Data_Entsoe/Data_Preprocessing/For_Multi_Step_Prediction/y_API.csv', delimiter=',')
y = y.drop(['Unnamed: 0'], axis=1)
y = y.iloc[-48*7*5:,-1]

# Split data into train set and test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle = False)

# x_scaler = StandardScaler()
# y_scaler = StandardScaler()
# X_train = x_scaler.fit_transform(X_train)
# X_test = x_scaler.transform(X_test)
# y_train = y_scaler.fit_transform(y_train)

epochs = 5000
learning_rate = 0.001
batches = 16

# Build the model.
model = build_model(X_train.shape[1],learning_rate)
# Run training session.
model.fit(X_train,y_train, epochs=epochs, batch_size=batches, verbose=False)
# Describe model.
model.summary()

# Plot the learning progress.
plt.plot(model.history.history["mean_absolute_error"][600:])
plt.show()

# Make a single prediction on the test set and plot.
prediction = model.predict(X_test)

#x_axis = np.linspace(1,len(prediction), len(prediction))
fig1, axs1=plt.subplots(1,1,figsize=(12,6))
axs1.plot(dates.iloc[-48*7:],prediction, label = "prediction", color = "blue")
axs1.plot(dates.iloc[-48*7:],y_test, label = "true", color = "red")
axs1.set_xlabel('Settlement Periods (Test Set)')
axs1.set_ylabel('Load [MW]')
axs1.legend()
loc = plticker.MultipleLocator(base=47) # this locator puts ticks at regular intervals
axs1.xaxis.set_major_locator(loc)
fig1.autofmt_xdate()
#axs1.xticks(np.arange(min(dates.iloc[-48*7:]), max(dates.iloc[-48*7:]), 10.0))
fig1.show()

# Make a 1000 predictions on the test set and calculate the errors for each prediction.
yhats = [model.predict(X_test) for _ in range(1000)]
predictions = np.array(yhats)
predictions = predictions.reshape(-1,len(predictions[1]))
# Make one large column vector containing all the predictions
predictions_vector = predictions.reshape(-1, 1)

# Create a np array with all the predictions in the first column and the corresponding error in the next column.
predictions_and_errors = np.zeros((len(predictions_vector),2))
predictions_and_errors[:,:-1] = predictions_vector
j=0
for i in range (len(predictions_vector)):
        predictions_and_errors[i,1] = predictions_vector[i]-y_test.iloc[j]
        j=j+1
        if j == len(y_test):
            j=0

# Calculate the stddev from the 1000 predictions.
mean = (sum(predictions)/1000).reshape(-1,1)
stddev = np.zeros((len(X_test),1))
for i in range (len(X_test)):
    stddev[i,0] = np.std(predictions[:,i])

mean_plus_stddev = mean+2*stddev
mean_minus_stddev = mean-2*stddev
# Plot the result with the truth in red and the predictions in blue.

fig2, axs2=plt.subplots(1,1,figsize=(12,6))
axs2.plot(dates.iloc[-48*7-48*3:-48*7],y_train[-48*3:], label = "Training Set", alpha = 1, color = "black")
axs2.plot(dates.iloc[-48*7:],mean,label = "Mean of the predictions", color = "blue")
# Potentially include all the predictions made
#fig2.plot(X_axis[-48*7:], predictions.T[:,:50], alpha = 0.1, color = "blue")
axs2.fill_between(dates.iloc[-48*7:], mean_plus_stddev.reshape(-1,), mean_minus_stddev.reshape(-1,), alpha=0.3, color = "blue")
axs2.plot(dates.iloc[-48*7:],y_test, label = "Test Set", alpha = 1, color = "red")
axs2.axvline(dates.iloc[-48*7], linestyle="--", color = "black")
axs2.set_xlabel('Settlement Periods (Test Set)')
axs2.set_ylabel('Load [MW]')
loc = plticker.MultipleLocator(base=47) # this locator puts ticks at regular intervals
axs2.xaxis.set_major_locator(loc)
fig2.autofmt_xdate()
axs2.legend()
fig2.show()

# Plot the histogram of the errors.
error_column = predictions_and_errors[:,1]
for i in range(60):
    error_column = np.delete(error_column,np.argmax(error_column),0)
    #error_column = np.delete(error_column,np.argmin(error_column),0)

plt.hist(error_column, bins = 25)
plt.xlabel("Prediction Error")
plt.ylabel("Count")
plt.show()
