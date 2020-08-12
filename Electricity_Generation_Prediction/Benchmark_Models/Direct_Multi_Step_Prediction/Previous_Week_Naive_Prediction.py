from matplotlib import pylab as plt
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.ticker as plticker

# Get the X (containing the features) and y (containing the labels) values
X = pd.read_csv('Data_Preprocessing/For_336_SP_Step_Prediction/X.csv', delimiter=',')
X = X.set_index("Time")
dates = X.iloc[:,-1]
X = X.iloc[:,:-1]

y = pd.read_csv('Data_Preprocessing/For_336_SP_Step_Prediction/y.csv', delimiter=',')
y = y.set_index("Time")

# Split data into train set and test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle = False)

X_train = X_train[int(len(X_train)*1/2):]
X_test = X_test[:int(len(X_test)*1/2)]
y_train = y_train[int(len(y_train)*1/2):]
y_test = y_test[:int(len(y_test)*1/2)]
dates = dates[-len(X_train)-len(X_test)*2:-len(X_test)]

# Plot the histograms
fig1, axs1=plt.subplots(1,1,figsize=(12,6))
axs1.grid(True)
error = (X.iloc[:,0]- y.iloc[:,0])/1000
axs1.hist(error, bins = 50, color = "blue")
axs1.set_xlabel("Error between the prediction (X) and the actual values (y) in GW", size = 14)
axs1.set_ylabel("Count", size = 14)
fig1.show()
fig1.savefig("Electricity_Generation_Prediction/Benchmark_Models/Figures/Previous_Week_Histogram_All.pdf", bbox_inches='tight')

mean = np.mean((X.iloc[:,0]- y.iloc[:,0]))
stddev = np.std((X.iloc[:,0]- y.iloc[:,0]))

# Print their mean and standard deviation
print("The mean of the errors is %.2f" %mean, "MW and the standard deviation is %.2f" % stddev,"MW." )

# Plot the histograms
fig3, axs3=plt.subplots(1,2,figsize=(12,6))
axs3[0].grid(True)
axs3[0].hist((X_train.iloc[:,0]- y_train.iloc[:,0])/1000, bins = 30, color = "blue")
axs3[0].set_xlabel("Error between the prediction (X_train)\nand the actual values (y_train) in GW", size = 14)
axs3[0].set_ylabel("Count", size = 14)

axs3[1].grid(True)
axs3[1].hist((X_test.iloc[:,0]- y_test.iloc[:,0])/1000, bins = 30, color = "blue")
axs3[1].set_xlabel("Error between the prediction (X_test)\nand the actual values (y_test) in GW", size = 14)
axs3[1].set_ylabel("Count", size = 14)
fig3.show()
fig3.savefig("Electricity_Generation_Prediction/Benchmark_Models/Figures/Previous_Week_Histograms_Train_and_Test.pdf", bbox_inches='tight')

mean_train = np.mean((X_train.iloc[:,0]- y_train.iloc[:,0]))
stddev_train = np.std((X_train.iloc[:,0]- y_train.iloc[:,0]))
mean_test = np.mean((X_test.iloc[:,0]- y_test.iloc[:,0]))
stddev_test = np.std((X_test.iloc[:,0]- y_test.iloc[:,0]))

# Print their mean and standard deviation
print("The mean of the training sets is %.2f" %mean_train, "MW and the standard deviation is %.2f" % stddev_train,"MW." )
print("The mean of the test set is %.2f" %mean_test,"MW and the standard deviation is %.2f" %stddev_test,"MW." )

# Naive prediction
pred_test = X_test.iloc[:,0]

error_test_plot = np.zeros((336+48*3,1))
error_test_plot[-336:,0] = X_test.iloc[:48*7,0]-y_test.iloc[:48*7,0]

# Plot the result with the truth in blue/black and the predictions in orange.
fig2, axs2=plt.subplots(2,1,figsize=(12,6))
axs2[0].grid(True)
axs2[0].plot(dates.iloc[-len(X_test)-48*3:-len(X_test)],
             y_train.iloc[-48*3:,0]/1000,
             label = "Training Set", alpha = 1, color = "blue")
axs2[0].plot(dates.iloc[-len(X_test):-len(X_test)+48*7],
             pred_test[:48*7]/1000,
             label = "Naive Prediction", color = "orange")
axs2[0].fill_between(dates.iloc[-len(X_test):-len(X_test)+48*7],
                     (pred_test[:48*7]+stddev_train)/1000,
                     (pred_test[:48*7]-stddev_train)/1000,
                     alpha=0.2, color = "orange", label = "+-1x\nStandard Deviation")
axs2[0].plot(dates.iloc[-len(X_test):-len(X_test)+48*7],
             y_test.iloc[:48*7,0]/1000,
             label = "Test Set", alpha = 1, color = "black")
axs2[0].axvline(dates.iloc[-len(X_test)], linestyle="--", color = "black")
axs2[0].set_ylabel('Load, GW',size = 14)
loc = plticker.MultipleLocator(base=47) # Puts ticks at regular intervals
axs2[0].xaxis.set_major_locator(loc)
axs2[0].plot(30,30,label = "Error", color = "red")

# plot the errors
axs2[1].grid(True)
axs2[1].plot(dates.iloc[-len(X_test)-48*3:-len(X_test)+48*7],
             error_test_plot/1000,
             label = "Error", alpha = 1, color = "red")
axs2[1].axvline(dates.iloc[-len(X_test)], linestyle="--", color = "black")
axs2[1].set_xlabel('2019',size = 14)
axs2[1].set_ylabel('Error, GW',size = 14)

# Include additional details such as tick intervals, rotation, legend positioning and grid on.
axs2[0].grid(True), axs2[1].grid(True)
loc = plticker.MultipleLocator(base=48) # Puts ticks at regular intervals
axs2[0].xaxis.set_major_locator(loc), axs2[1].xaxis.set_major_locator(loc)
fig2.autofmt_xdate(rotation=0)
axs2[0].legend(loc=(1.02,0.49)),

plt.xticks(np.arange(1,482, 48), ["14:00\n07/22","14:00\n07/23","14:00\n07/24",
                                  "14:00\n07/25","14:00\n07/26","14:00\n07/27",
                                  "14:00\n07/28","14:00\n07/29","14:00\n07/30",
                                  "14:00\n07/31","14:00\n08/01"])

fig2.show()
fig2.savefig("Electricity_Generation_Prediction/Benchmark_Models/Figures/Previous_Week_Prediction.pdf", bbox_inches='tight')

# Calculate the errors from the mean to the actual vaules.
error_train = (X_train.iloc[:,0]-y_train.iloc[:,0])/1000
error_test = (X_test.iloc[:,0]-y_test.iloc[:,0])/1000

print("-"*200)
print("The mean absolute error of the entire test set is %0.2f" % np.mean(abs(error_test)))
print("The mean squared error of the entire test set is %0.2f" % np.mean(error_test**2))
print("The root mean squared error of the entire test set is %0.2f" % np.sqrt(np.mean(error_test**2)))
print("-"*200)
print("The mean absolute error of the entire training set is %0.2f" % np.mean(abs(error_train)))
print("The mean squared error of the entire training set is %0.2f" % np.mean(error_train**2))
print("The root mean squared error of the entire training set is %0.2f" % np.sqrt(np.mean(error_train**2)))
print("-"*200)

########################################################################################################################
# Save the results in a csv file.
########################################################################################################################

df_errors = pd.DataFrame({"MSE_Train": [np.mean(error_train**2)],
                          "MAE_Train": [np.mean(abs(error_train))],
                          "RMSE_Train": [np.sqrt(np.mean(error_train**2))],
                          "MSE_Test": [np.mean(error_test**2)],
                          "MAE_Test": [np.mean(abs(error_test))],
                          "RMSE_Test": [np.sqrt(np.mean(error_test**2))],
                          })
df_errors.to_csv("Compare_Models/Direct_Multi_Step_Results/Naive.csv")
df_errors.to_csv("Compare_Models/Direct_Multi_Step_Probability_Results/Probability_Based_on_Training/Naive_error.csv")

