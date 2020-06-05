# Prediciting the temperature using a Recurrent Neural Network (LSTM to be more precise)
# Part 1 - Data Preprocessing
# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.optimizers import RMSprop

# Importing the training set
df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv')
# Transform the date in datetime type
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date")

# Extract the year, month and day of the date of each observation (1 temp. per day).
df["Year"] = df.index.year
df["Month"] = df.index.month
df["Day"] = df.index.day

# Reset the index and drop the date column.
df.reset_index(inplace = True)
df = df.drop(columns = ["Date"])

# Split the dataframe in training and testing set.
nmb_rows_df = df.shape[0]
index = int(0.8*nmb_rows_df)
train_df = df[0:index].copy()
test_df = df[index:].copy()

# Define the NN input features: Year, Month, Day and Temperature.
# The temperature of the nex day will be predicted with the temperature from the 30 previous days.
Raw_Input = train_df.iloc[:, 0:4].values
Raw_Output = train_df[["Temp"]].values

# Feature Scaling leaving it out for now. #todo Should I include this later?
from sklearn.preprocessing import MinMaxScaler
# Feature range equals (0, 1), that's the default feature range
# But also looking at the formula, Normalization gives you
# a range between 0 and 1.
#scaler = MinMaxScaler(feature_range = (0, 1))
# fit_transform is a method of the MinMaxScaler. It will fit the object scaler to the training and it will transform it (scale it).
# More specifically, fit will get the minimum and maximum of the data.


# Creating a data structure with 30 previous observations and 1 output
X_train = []
y_train = []

X_train = np.zeros((4,len(Raw_Input)-30,30))
y_train = np.zeros((len(Raw_Input)-30,1))
for j in range(4):
    for i in range(30, len(Raw_Input)):
        for k in range(30):
            X_train[j, i - 30, k] = Raw_Input[i-k-1, j]
            y_train[i - 30] = Raw_Input[i,0]

# Reshaping
# Keras requires an input of 3 dimensions
# The first is the number of observations
# The second is the number of observations we pass in from the past
# The third enables the dates to be passed in as well.
X_train = np.reshape(X_train, (X_train.shape[1], X_train.shape[2], X_train.shape[0]))

# X_train = scaler.fit_transform(X_train)
# y_train = scaler.fit_transform(y_train)

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
# Initialise the RNN as a sequence of layers as opposed to a computational graph.
regressor = Sequential()

# Because predicting the temperature is pretty complex, you need to have a high dimensionality too thus 50
# for the number of neurons. If the number of neurons is too small in each of the LSTM layers, the model would not
# capture very well the upward and downward trend.
regressor.add(LSTM(units = 25, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
regressor.add(Dropout(0.1))

# Adding a second LSTM layer and Dropout regularisation
# No need to specify any input shape here because we have already defined that we have 50 neurons in the
# previous layer.
regressor.add(LSTM(units = 25, return_sequences = True))
regressor.add(Dropout(0.1))

# Adding a third LSTM layer and Dropout regularisation
regressor.add(LSTM(units = 25 ))
regressor.add(Dropout(0.1))

# # Adding a fourth LSTM layer and Dropout regularisation
# # This is the last LSTM layer that is  added! Thus the return sequences is set to  false.
# regressor.add(LSTM(units = 25))
# regressor.add(Dropout(0.1))

# Adding the output layer
# We are not adding an LSTM layer. We are fully connecting the outward layer to the previous LSTM layer.
# As a result, we use a DENSE layer to make this full connection.
regressor.add(Dense(units = 1))

# Compiling the RNN
# For RNN and also in the Keras documentation, an RMSprop is recommended.
# But experimenting with other optimizers, one can also use the adam optimizer.
# The adam optimizer is actually always a good choice and very powerfull too!
# In general, the most commonly used optimizers are adam and RMSprop
optimizer = RMSprop(lr = 1e-3)
regressor.compile(optimizer = optimizer, loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 20, batch_size = 64)

# Part 3 - Making the predictions and visualising the results

# Getting the real temperatures.
test_temperatures = test_df.values

# For the test set, we also need the temperatures of the 30 previous days.
# As a result, we need to concatinate the training and testing set.
dataset_total = pd.concat((train_df, test_df), axis = 0)

# To get the input values for the prediciton, think about the lower and upper bounds. For the first day
# in the test set, you start at the bottom len(dataset_total) (which is the last day of the test set)
# and then you substract the number of days in the test set len(dataset_test).
# The first day requires the 30 previous days too (as an input) which is why you need to start even earlier and
# substract these 30 too. The value we just calculated is the observation from which you want to start.
# From this value onward and all the values further down (:) will be fed into the model to create a prediciton.
# .values is just to make this a numpy array.
inputs = dataset_total[len(dataset_total) - len(test_df) - 30:].values

# Creating the 3D array for the test set similarly as before.
X_test= np.zeros((4,len(inputs)-30,30))
for j in range(4):
    for i in range(30, len(inputs)):
        for k in range(30):
            X_test[j, i - 30, k] = inputs[i-k-1, j]

# # Just as above, reshape the input before it goes into the NN.
X_test = np.reshape(X_test, (X_test.shape[1], X_test.shape[2], X_test.shape[0]))
# # Predict the stock prices.
#X_test = scaler.fit_transform(X_test)

# # Rescale the output.
predicted_temperature = regressor.predict(X_test)
# predicted_temperature = scaler.inverse_transform(predicted_temperature)


# Visualising the results
error = predicted_temperature - test_temperatures

plt.plot(test_temperatures[:,0], color = 'red', label = 'Real Temperature')
plt.plot(predicted_temperature[:,0], color = 'blue', label = 'Predicted Temperature')
plt.plot(abs(error[:,0]), color = 'black', label = 'Error')
plt.title('Temperature Prediction')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.legend()
plt.show()

print("The absolute men error is ", sum(abs(error))/len(predicted_temperature))