import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from Functions_ANN import plot_the_loss_curve, train_model, previousday, nested_sum, create_model

########################################################################################################################

# Get data and data preprocessing.

########################################################################################################################

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

########################################################################################################################

# Create the model.

########################################################################################################################

# Define the hyperparameters.
learning_rate = 0.0005
epochs = 100
batch_size = 32

# Create the model.
my_model = create_model(learning_rate, (4,))

# Extract the loss per epoch to plot the learning progress.
epochs, loss = train_model(my_model, NN_Input, NN_Output_Labels, epochs, batch_size)

# Plot the loss per epoch.
plot_the_loss_curve(epochs[10:], loss[10:])

########################################################################################################################

# Data processing for predicting the temperatures.

########################################################################################################################

# Just as before, the test_df is also used to shift the temperature input by 1 day.
# Created a dataframe compare_temp_df containing the original temperature (not shifted) and the modified
# temperature shifted by one day which is used as an input to the NN to test the prediciton.
compare_temp_df = test_df.copy()
compare_temp_df = compare_temp_df.drop(columns = ["Date", "Year", "Month", "Day"])
compare_temp_df = compare_temp_df.rename(columns = {"Temp": "Temp_before"})

# Shift the temperature of the test_df dataframe by 1 day.
# Add the shifted column to the dataframe compare_temp_df.
test_df['Temp'] = test_df['Temp'].shift(periods = 1, fill_value = test_df['Temp'].mean())
compare_temp_df['Temp_after'] = test_df['Temp']

# Predict Temperature with the NN using the test data.
# Use the temperature form the previous day.
test_values = test_df.drop(columns = ["Date"]).copy()

# predicted_NN_temperatures = my_model.predict(test_values)
arr = np.array
for counter in range(0,len(test_values)):
    row = test_values.loc[2920 + counter, :]
    row_values = np.reshape(row.values, (1, 4))
    predicted_NN_temperatures = my_model.predict(row_values)
    arr = np.append(arr, predicted_NN_temperatures)

########################################################################################################################

# Data processing for plotting curves and printing the errors.

########################################################################################################################

dates_of_predicted_NN_temperatures = test_df['Date']

# Create the Previous Day Dataframe (which is the original dataframe but shifted by a day)
predicted_df = previousday(df.copy())
error_previousday = abs(predicted_df['Temp'] - df['Temp'])
print("The mean absolute error from the previous day prediciton is %.2f" %np.average(error_previousday))
print("The mean squarred error from the previous day prediciton is %.2f" %np.average(error_previousday*error_previousday))

# Turn the dataframe into a Numpy Array.
given_temp_array = df['Temp'][index:].to_numpy()

# Reshape the Arrays to substract one array from the other.
given_temp_array = given_temp_array.reshape(test_df.shape[0],1)

# Compute the error between the Actual Temperature and the
test = arr[1:]
error_NN = abs(test.reshape(730,1) - given_temp_array)
print("The mean absolute error from the NN prediciton is %.2f"  %np.mean(error_NN))
print("The mean squarred error from the NN prediciton is %.2f"  %np.mean(error_NN*error_NN))

########################################################################################################################

# Plotting curves.

########################################################################################################################

# Plot the actual recorded temperature against the date.
from Functions_ANN import plot_actual_temp, plot_predicted_temp, plot_error, plot_prediction_zoomed_in

fig, axes = plt.subplots(3)

# Plot the actual temperatures in a subplot of 3x1.
plot_actual_temp(axes, df)

# Plot the predicted temperature (from the day before) against the recording date.
plot_predicted_temp(axes, predicted_df["Date"].values, predicted_df["Temp"].values, "Previous day prediciton")

# Plot the error between the previous day and the actual temperature.
plot_error(axes, predicted_df["Date"], error_previousday, "NN error")

# Plot the actual recorded temperature against the date.
fig2, axes2 = plt.subplots(3)

# Plot the actual temperatures in a new subplot of 3x1.
plot_actual_temp(axes2, df[index:])

# Plot the the predicted (NN) temperature.
plot_predicted_temp(axes2, dates_of_predicted_NN_temperatures, arr[1:], "NN prediciton")

# Plot the error between the predicted and the actual temperature.
plot_error(axes2, dates_of_predicted_NN_temperatures, error_NN, "NN error")

# Plot the predicted temperature on the last 60 days.
plot_prediction_zoomed_in(dates_of_predicted_NN_temperatures[-60:], arr[-60:],df['Temp'][-60:], predicted_df["Temp"].values[-60:], "Predicted", "Actual", "Previous day")

