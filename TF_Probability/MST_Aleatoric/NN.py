from matplotlib import pylab as plt
import numpy as np
from sklearn.model_selection import train_test_split
from numpy import genfromtxt
from TF_Probability.MST_2.Functions import build_model
from sklearn.preprocessing import StandardScaler
from tensorflow_probability import distributions as tfd
import pandas as pd


# Get the X (containing the features) and y (containing the labels) values
# X = genfromtxt('Data_Preprocessing/For_336_SP_Step_Prediction/X.csv', delimiter=',')
# X = X[:48*7*5,:]

y = genfromtxt('Data_Preprocessing/For_336_SP_Step_Prediction/y.csv', delimiter=',')
dates = y.iloc[8760:69277,1]
y = y.iloc[8760:69277,1]

projected_data = pd.read_csv('TF_Probability/MST_Aleatoric/Projected_Data', delimiter=',')
projected_data = projected_data.iloc[8760:69277,1:]

# # Split data into train set and test set.
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle = False)

# x_scaler = StandardScaler()
# y_scaler = StandardScaler()
# X_train = x_scaler.fit_transform(X_train)
# X_test = x_scaler.transform(X_test)
# y_train = y_scaler.fit_transform(y_train)

epochs = 2500
learning_rate = 0.001
batches = 16

# Build the model.
model = build_model(learning_rate)
# Run training session.
model.fit(projected_data,y, epochs=epochs, batch_size=batches, verbose=False)
# Describe model.
model.summary()

# Plot the learning progress.
plt.plot(model.history.history["mean_absolute_error"])
plt.show()

# Make a single prediction on the test set and plot.
prediction = model(X_test)
#assert isinstance(prediction, tfd.Distribution)

fig1 = plt
fig1.plot(prediction, label = "prediction", color = "blue")
fig1.plot(y_test, label = "true", color = "red")
fig1.xlabel('Settlement Periods (Test Set)')
fig1.ylabel('Load [MW]')
fig1.legend()
fig1.show()

# Make a 100 predictions on the test set and plot.
yhats = [model.predict(X_test) for _ in range(100)]
predictions = np.array(yhats)
predictions = predictions.reshape(-1,len(predictions[1]))
predictions_vector = predictions.reshape(-1, 1)

predictions_and_errors = np.zeros((len(predictions_vector),2))
predictions_and_errors[:,:-1] = predictions_vector

mean = (sum(predictions)/100).reshape(len(X_test),1)
stdev = np.zeros((len(X_test),1))

for i in range(48*7):
    stdev[i,0] = np.std(predictions[:,i])
dates = np.linspace(1,48*7,48*7)
plt.plot(dates, mean,color = "blue")
plt.plot(dates, y_test, color = "red")
#plt.plot(predictions.T, alpha = 0.1, color = "blue")
plt.fill_between(dates,  mean[:,0] - 2*stdev[:,0], mean[:,0] + 2*stdev[:,0], alpha=0.1, color = "blue")
plt.show()

j=0
for i in range (len(predictions_vector)):
        predictions_and_errors[i,1] = predictions_vector[i]-y_test[j]
        j=j+1
        if j == len(y_test):
            j=0

fig2=plt
fig2.plot(predictions[1,:].T,label = "prediction", alpha = 0.1, color = "blue")
fig2.plot(predictions[1:,:].T, alpha = 0.1, color = "blue")
fig2.plot(y_test, label = "true", alpha = 1, color = "red")
fig2.xlabel('Settlement Periods (Test Set)')
fig2.ylabel('Load [MW]')
fig2.legend()
fig2.show()

# Plot the histogram of the errors.
plt.hist(predictions_and_errors[:,1], bins = 25)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")
plt.show()

plt.plot(X_train[:,0])
plt.xlabel("SP")
plt.ylabel("Load [MW]")
plt.show()

# Calculate the errors
print("-"*200)
errors = abs(y_test-mean)
print("The mean absolute error of the test set is %0.2f" % np.mean(errors))
print("The mean squared error of the test set is %0.2f" % np.mean(errors**2))
print("The root mean squared error of the test set is %0.2f" % np.sqrt(np.mean(errors**2)))
print("-"*200)


########################################################################################################################
# Save the results in a csv file.
########################################################################################################################

import csv
with open('TF_Probability/Results/NN_mean_error.csv', 'w', newline='', ) as file:
    writer = csv.writer(file)
    writer.writerow(["Method","MSE","MAE","RMSE"])
    writer.writerow(["NN",
                     str(np.mean(errors**2)),
                     str(np.mean(errors)),
                     str(np.sqrt(np.mean(errors**2)))
                     ])
