import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the results of the different models in respective variables.
NN_error = pd.read_csv("Compare_Models/SMST_Probability_results/Probability_Based_on_Model/NN_error.csv")
NN_mean_stddev = pd.read_csv("Compare_Models/SMST_Probability_results/Probability_Based_on_Model/NN_mean_errors_stddevs.csv")
SARIMA_error = pd.read_csv("Compare_Models/SMST_Probability_results/Probability_Based_on_Model/SARIMA_error.csv")
SARIMA_mean_stddev = pd.read_csv("Compare_Models/SMST_Probability_results/Probability_Based_on_Model/SARIMA_mean_errors_stddevs.csv")


# Load the results of the different models in a dataframe.
frames = ([NN_error,SARIMA_error])
df = pd.concat(frames, axis = 0)
string = (['NN', 'SARIMA'])

# Create histograms for RMSE, MSE and MAE.
fig2, axes2 = plt.subplots(1,3,figsize=(12,6))
axes2[0].bar(df.iloc[:,0], df.iloc[:,1], color='blue')
axes2[0].set_ylabel('MSE [GW^2]', size = 14)
axes2[0].set_xticklabels(rotation=0, labels = string)
axes2[0].grid(True)

axes2[1].bar(df.iloc[:,0], df.iloc[:,2], color='blue')
axes2[1].set_ylabel('MAE [GW]', size = 14)
axes2[1].set_xticklabels(rotation=0, labels = string)
axes2[1].grid(True)

axes2[2].bar(df.iloc[:,0], df.iloc[:,3], color='blue')
axes2[2].set_ylabel('RMSE [GW]', size = 14)
axes2[2].grid(True)
axes2[2].set_xticklabels(rotation=0, labels = string)
fig2.show()

mean_error_NN = NN_mean_stddev.iloc[:,0] - NN_mean_stddev.iloc[:,2]
stddev_NN = NN_mean_stddev.iloc[:,1]

mean_error_SARIMA = SARIMA_mean_stddev.iloc[:,1]
stddev_SARIMA = SARIMA_mean_stddev.iloc[:,2]/1000

x_axis = np.linspace(1,336,336)

# Plot the mean and standard deviations.
fig3, axes3 = plt.subplots(2,1,figsize=(12,6))
axes3[0].plot(x_axis, mean_error_NN[:48*7], label= "Mean Error", color='orange')
axes3[0].fill_between(x_axis,
                      (mean_error_NN+stddev_NN)[:48*7],
                      (mean_error_NN-stddev_NN)[:48*7],
                      label= "Standard Deviation" ,alpha = 0.2, color='orange')
axes3[0].set_ylabel('Mean Error', size = 14)
axes3[0].set_xlabel("Settlement Periods", size = 14)
axes3[0].grid(True)

axes3[1].plot(x_axis, mean_error_SARIMA, label= "Mean Error", color='orange')
axes3[1].fill_between(x_axis,
                      (mean_error_SARIMA+stddev_SARIMA)[:48*7],
                      (mean_error_SARIMA-stddev_SARIMA)[:48*7],
                      label= "Standard Deviation" ,alpha = 0.2, color='orange')
axes3[1].set_ylabel('Mean Error', size = 14)
axes3[1].set_xlabel("Settlement Periods", size = 14)
axes3[1].grid(True)
fig3.show()
