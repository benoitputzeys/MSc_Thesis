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

# Creating a data structure with 30 previous observations and 1 output
X_train = []
y_train = []
days_past = 1

X_train = np.zeros((len(Raw_Input)-days_past,days_past,4))
y_train = np.zeros((len(Raw_Input)-days_past,1))
for j in range(4):
    for i in range(days_past, len(Raw_Input)):
        for k in range(days_past):
            X_train[i - days_past, k, j] = Raw_Input[i-k-1, j]
            y_train[i - days_past] = Raw_Input[i,0]

# Part 2 - Building the RNN
from Functions_LSTM import train_model, plot_loss, create_model

regressor = create_model(X_train)

# Extract the loss per epoch and plot it.
epochs, mse = train_model(regressor, X_train, y_train)
plot_loss(epochs, mse)

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
inputs = dataset_total[len(dataset_total) - len(test_df) - days_past:].values

# Creating the 3D array for the test set similarly as before.
X_test= np.zeros((len(inputs)-days_past,days_past,4))
for j in range(4):
    for i in range(days_past, len(inputs)):
        for k in range(days_past):
            X_test[i - days_past, k, j] = inputs[i-k-1, j]


# Predict the temperature with the test values.
predicted_temperature = regressor.predict(X_test)

# Compute the error.
error = predicted_temperature - test_temperatures

# Visualise the results.
plt.figure(2)
plt.plot(test_temperatures[:,0], color = 'red', label = 'Real Temperature')
plt.plot(predicted_temperature[:,0], color = 'blue', label = 'Predicted Temperature')
plt.plot(abs(error[:,0]), color = 'black', label = 'Error')
plt.title('Temperature Prediction')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.legend()
plt.show()

print("The absolute men error is", sum(abs(error))/len(predicted_temperature))