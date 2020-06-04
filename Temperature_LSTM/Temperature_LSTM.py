# Code here.
# Recurrent Neural Network
# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv')
# Transform the date in datetime type
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date")

# Extract the year, month and day of the date of each observation (1 temp. per day).
df["Year"] = df.index.year
df["Month"] = df.index.month
df["Day"] = df.index.day

# Reset the index.
df.reset_index(inplace = True)

# Split the dataframe in training and testing set.
nmb_rows_df = df.shape[0]
index = int(0.8*nmb_rows_df)
train_df = df[0:index].copy()
test_df = df[index:].copy()

# Define the NN input features: Year, Month, Day and Temperature: Drop the Date columns.
# As the temperature of the nex day will be predicted with the temperature from the previous day, the NN should also
# be trained with the previous day temperatures.
train_shifted_df = train_df.copy()
train_shifted_df['Temp'] = train_shifted_df['Temp'].shift(periods = 1, fill_value = 0)
train_shifted_df = train_shifted_df.drop(columns = ["Date"])
Raw_Input = train_shifted_df.iloc[:, 0:4].values
Raw_Output = train_df[["Temp"]].values

# Feature Scaling leaving it out for now. #todo Should I include this later?
# from sklearn.preprocessing import MinMaxScaler
# # Feature range equals (0, 1), that's the default feature range
# # But also looking at the formula, Normalization gives you
# # a range between 0 and 1.
# scaler = MinMaxScaler(feature_range = (0, 1))
# # fit_transform is a method of the MinMaxScaler. It will fit the object scaler to the training and it will transform it (scale it).
# # More specifically, fit will get the minimum and maximum of the data.
# scaled_training_set = scaler.fit_transform(Raw_Input)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(30, len(Raw_Input)):
    X_train.append(Raw_Input[i-30:i, 0])
    y_train.append(Raw_Input[i, 0])
# Remember that in order to be accepted into the network, you need to transform it into a numpy array.
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
# Keras requires an input of 3 dimensions
# The first is the number of observations
# The second is the number of timesteps
# The third is the number of indicators.
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
# Initialise the RNN as a sequence of layers as opposed to a computational graph.
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
# We need to input 3 arguments:
# 1st, the number of units which is the number of LSTM cells or memory units
# you want to have in this LSTM layer.
# The 2nd input is return sequences, as we have more than 1 LSTM layer, you
# have to set the retunr sequence argument to true. In the last LSTM layer,
# you can omit this step because the default parameter is false which is what we want.
# The 3rd and lat argument is the input shape which is the shape of the input containing x_train
# It has an input shape of 3 dimensions corresponding to the obervations, the timesteps and
# the indicators. We do not need to include the first dimension because the
# number of observations will already be taken into account.
# Remember the last 1 in the input_shape is the numbe rf indicators.

# Because predicting the stock market price is pretty complex, you need to have a high dimensionality too thus 50
# for the number of neurons. If the number of neurons is too small in each of the LSTM layers, the model would not
# capture very well the upward and downward trend.
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
# No need to specify any input shape here because we have already defined that we have 50 neurons in the
# previous layer.
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
# This is the last LSTM layer that you are adding! Thus you need to say that the return sequences
# are false. You are not going to return any more seuqences.
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
# We are not adding an LSTM layer! We are fully connecting the outward layer to the previous LSTM layer.
# As a result, we use a DENSE layer to make this full connection.
regressor.add(Dense(units = 1))

# Compiling the RNN
# For RNN and also in the Keras documentation, an RMSprop is recommended.
# But experimenting with other optimizers, one can also use the adam optimizer.
# The adam optimizer is actually always a good choice and very powerfull too!
# In general, the most commonly used optimizers are adam and RMSprop
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
# Remember number of epochs is the number of times you want your whole training dataset to
# be used in forward propagation and backpropagated to update the weights.
# Also instead of updating the weights for every observation, update the weights after
# 32 stock prize observation.
regressor.fit(X_train, y_train, epochs = 45, batch_size = 32)



# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
test_temperatures = test_df[["Temp"]].values

# Getting the predicted stock price of 2017
# For the test set, we also need the google stock market price of the 60 previous days.
# As a result, we need to concatinate the training and testing set.
# One thing we should never do is to modify the actual test values. They have to remain how they are
# Thus make sure to define a new variable containing the modified data and work with those.
# With this new variable, you can scale it so that it corresponds to what the algorithm has
# learned on. This new variable prohibits you from changing the test values.
dataset_total = pd.concat((train_df['Temp'], test_df['Temp']), axis = 0)
# Sidenote, you are training on 3 months because the google market value is given for each working day.
# 60 working days / 5 is 12. This /4 is 3 months.
# The way he got this was to think about the lower and upper bounds. To get the first day
# in the test set, you start at the bottom len(dataset_total) (which is the last day of the test set)
# and then you substract the number of days in the test set len(dataset_test).
# The first day requires the 60 previous days as well as an input which is why you need to
# start even earlier and substract these 60 too. The value you just calculated is
# the observation from which you want to start. From this value onward and all
# the values further down (:) will be fed into your model to create a prediciton.
# .values is just to make this a numpy array.
inputs = dataset_total[len(dataset_total) - len(test_df) - 30:].values
# The shape of inputs is (80,) but sklearn requires this to be in a shape (80,1)
# Thus just put a reshape and all good.
inputs = inputs.reshape(-1,1)
# # As you have trained on reshaped values, you have to reshape the inputs for the prediciton as well.
# # Careful here to not use fit_transform as above because you do not want to compute the minimum and maximum
# # again, you want to preserve those from before and subject your data to the same tranformation as before.
# inputs = sc.transform(inputs)

# Remember that in this step, you create a numpy array with 60 entries per rowand 20 columns
# In each iteration, you add this row to the variable X_test via append.
# After this, make sure to also make it an np.array.
X_test = []
for i in range(30, len(inputs)):
    X_test.append(inputs[i-30:i, 0])
X_test = np.array(X_test)

# Just as above, reshape your input before it goes into the NN.
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# Predict the stock prices.
predicted_temperature = regressor.predict(X_test)
# # Rescale the output.
# predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(test_temperatures, color = 'red', label = 'Real Temperature')
plt.plot(predicted_temperature, color = 'blue', label = 'Predicted Temperature')
plt.title('Temperature Prediction')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.legend()
plt.show()
