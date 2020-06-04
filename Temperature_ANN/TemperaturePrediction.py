import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import keras

# Import the timeseries data
df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv")

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
NN_Input = train_shifted_df.drop(columns = ["Date"])
NN_Output_Labels = train_df[["Temp"]]

# Define the hyperparameters.
learning_rate = 0.0005
epochs = 150
batch_size = 40

from Functions import plot_the_loss_curve, train_model, previousday, nested_sum

# Create the model.
my_model = keras.Sequential()

# Input shape corresponds to the number of columns (the features day, month and year) of the dataframe,
# excpet for the output label, the temperature.

my_model.add(keras.layers.Dense(50, kernel_initializer = 'uniform', activation = 'relu', input_shape = (4,)))
# todo Exercise from Steve, try and learn all about the very basic stuff, how the NN is being constructed and how it will look like with the neurons.
# For example what happens if I put all the weights and biases to 1? What is the output etc. ?
# todo hold back some data and do the exercise Steve gave you in Slack. Make sure to pose the problem right:
# If you do not give the NN a chance to learn smth about the year then it cannot do that.
my_model.add(keras.layers.Dense(85, kernel_initializer = 'uniform', activation = 'relu'))
my_model.add(keras.layers.Dense(85, kernel_initializer = 'uniform', activation = 'relu'))
my_model.add(keras.layers.Dense(1, kernel_initializer = 'uniform', activation = 'relu'))
# todo get the accuracy of the training set and the testing set.
opt = keras.optimizers.Adam(learning_rate)
my_model.compile(loss = 'mean_squared_error', optimizer = opt)

# Extract the loss per epoch to plot the learning progress.
epochs, loss = train_model(my_model, NN_Input, NN_Output_Labels, epochs, batch_size)
plot_the_loss_curve(epochs, loss)

# Just as before, the test_df is also used to shift the temperature input by 1 day.
# Created a dataframe compare_temp_df containing the original temperature (not shifted) and the modified
# temperature shifted by one day which is used as an input to the NN to test the prediciton.
compare_temp_df = test_df.copy()
compare_temp_df = compare_temp_df.drop(columns = ["Date", "Year", "Month", "Day"])
compare_temp_df = compare_temp_df.rename(columns = {"Temp": "Temp_before"})
# Shift the temperature of the test_df dataframe by 1 day.
test_df['Temp'] = test_df['Temp'].shift(periods = 1, fill_value = test_df['Temp'].mean())
# Add the shifted column to the dataframe compare_temp_df.
compare_temp_df['Temp_after'] = test_df['Temp']
print(compare_temp_df)

# Predict Temperature with the NN using the test data.
# Use the temperature form the previous day.
test_values = test_df.drop(columns = ["Date"]).copy()
predicted_NN_temperatures = my_model.predict(test_values)
dates_of_predicted_NN_temperatures = test_df['Date']

# Plot the actual recorded temperature against the date.
fig, ax=plt.subplots(3)
ax[0].plot(df['Date'], df['Temp'], linewidth = 0.5)
ax[0].set_xlabel("Years")
ax[0].set_ylabel("Actual Temperature")

# Create the Previous Day Dataframe (which is the original dataframe but shifted by a day)
predicted_df = previousday(df.copy())
error = abs(predicted_df['Temp'] - df['Temp'])
print("The mean absolute error from the previous day prediciton is %.2f" %nested_sum(error))
print("The mean squarred error from the previous day prediciton is %.2f" %nested_sum(error*error))

# Plot the predicted temperature (from the day before) against the recording date.
ax[1].plot(predicted_df['Date'], predicted_df['Temp'], linewidth = 0.5)
ax[1].set_xlabel("Years")
ax[1].set_ylabel("Previous Day Temperature")

# Plot the error between the predicted temperature (obtained from the previous day) and the actual temperature.
ax[2].plot(predicted_df['Date'], error, linewidth = 0.5)
ax[2].set_xlabel("Years")
ax[2].set_ylabel("Previous Day Error")

plt.show()

# Plot the actual recorded temperature against the date.
fig2, ax2=plt.subplots(3)

ax2[0].plot(test_df['Date'], df['Temp'][index:], linewidth = 0.5)
ax2[0].set_xlabel("Years")
ax2[0].set_ylabel("Actual Temperature")

ax2[1].plot(dates_of_predicted_NN_temperatures, predicted_NN_temperatures, linewidth = 0.5)
ax2[1].set_xlabel("Years")
ax2[1].set_ylabel("NN Predicted Temperature")

# Turn the dataframe into a Numpy Array.
given_temp_array = df['Temp'][index:].to_numpy()

# Reshape the Arrays to substract one array from the other.
nmb_rows_test_df = test_df.shape[0]
given_temp_array = given_temp_array.reshape(nmb_rows_test_df,1)
predicted_NN_temperatures = predicted_NN_temperatures.reshape(nmb_rows_test_df,1)

# Compute the error between the Actual Temperature and the
error = abs(predicted_NN_temperatures - given_temp_array)
print("The mean absolute error from the NN prediciton is %.2f" %nested_sum(error)[0])
print("The mean squarred error from the NN prediciton is %.2f" %nested_sum(error*error)[0])
# Plot the error between the predicted temperature (obtained from the previous day) and the actual temperature.
ax2[2].plot(dates_of_predicted_NN_temperatures, error, linewidth=0.5)
ax2[2].set_xlabel("Years")
ax2[2].set_ylabel("NN Error")

plt.show()
