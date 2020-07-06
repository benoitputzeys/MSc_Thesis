from matplotlib import pylab as plt
import numpy as np
from sklearn.model_selection import train_test_split
from numpy import genfromtxt
from TF_Probability.Functions import build_model
from sklearn.preprocessing import StandardScaler

# Get the X (containing the features) and y (containing the labels) values
X = genfromtxt('Data_Entsoe/Data_Preprocessing/X.csv', delimiter=',')
X = X[:48*8,:]

y = genfromtxt('Data_Entsoe/Data_Preprocessing/y.csv', delimiter=',')
y = np.reshape(y, (len(y), 1))
y = y[:48*8]

# Split data into train set and test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle = False)

x_scaler = StandardScaler()
y_scaler = StandardScaler()
X_train = x_scaler.fit_transform(X_train)
X_test = x_scaler.transform(X_test)
y_train = y_scaler.fit_transform(y_train)

epochs = 5000
learning_rate = 0.001
batches = 16

# Build the model.
model = build_model(X_train,learning_rate)
# Run training session.
model.fit(X_train,y_train, epochs=epochs, batch_size=batches, verbose=False)
# Describe model.
model.summary()

# Plot the learning progress.
plt.plot(model.history.history["mean_absolute_error"])
plt.show()

# Make a single prediction on the test set and plot.
prediction = y_scaler.inverse_transform(model.predict(X_test))

fig1 = plt
fig1.plot(prediction, label = "prediction", color = "blue")
fig1.plot(y_test, label = "true", color = "red")
fig1.xlabel('Settlement Periods (Test Set)')
fig1.ylabel('Load [MW]')
fig1.legend()
fig1.show()

# Make a 100 predictions on the test set and plot.
yhats = [y_scaler.inverse_transform(model.predict(X_test)) for _ in range(100)]
predictions = np.array(yhats)
predictions = predictions.reshape(-1,len(predictions[1]))
predictions_vector = predictions.reshape(-1, 1)

predictions_and_errors = np.zeros((len(predictions_vector),2))
predictions_and_errors[:,:-1] = predictions_vector

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

plt.plot(x_scaler.inverse_transform(X_train[:,0]))
plt.xlabel("SP")
plt.ylabel("Load [MW]")
plt.show()