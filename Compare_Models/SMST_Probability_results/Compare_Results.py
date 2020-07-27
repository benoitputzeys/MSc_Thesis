import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the results of the different models in respective variables.
# Probability based on the models
NN_Rd_weigths_error = pd.read_csv("Compare_Models/SMST_Probability_results/Probability_Based_on_Model/NN_error.csv")
NN_Rd_weigths_mean_stddev = pd.read_csv("Compare_Models/SMST_Probability_results/Probability_Based_on_Model/NN_mean_errors_stddevs.csv")
SARIMA_error = pd.read_csv("Compare_Models/SMST_Probability_results/Probability_Based_on_Model/SARIMA_error.csv")
SARIMA_mean_stddev = pd.read_csv("Compare_Models/SMST_Probability_results/Probability_Based_on_Model/SARIMA_mean_errors_stddevs.csv")

# Probability based on the training
NN_error = pd.read_csv("Compare_Models/SMST_Probability_results/Probability_Based_on_Training/NN_error.csv")
NN_mean_stddev = pd.read_csv("Compare_Models/SMST_Probability_results/Probability_Based_on_Training/NN_mean_errors_stddevs.csv")
LSTM_error = pd.read_csv("Compare_Models/SMST_Probability_results/Probability_Based_on_Training/LSTM_error.csv")
LSTM_mean_stddev = pd.read_csv("Compare_Models/SMST_Probability_results/Probability_Based_on_Training/LSTM_mean_errors_stddevs.csv")
RF_error = pd.read_csv("Compare_Models/SMST_Probability_results/Probability_Based_on_Training/RF_error.csv")
RF_mean_stddev = pd.read_csv("Compare_Models/SMST_Probability_results/Probability_Based_on_Training/RF_mean_errors_stddevs.csv")
SVR_error = pd.read_csv("Compare_Models/SMST_Probability_results/Probability_Based_on_Training/SVR_error.csv")
SVR_mean_stddev = pd.read_csv("Compare_Models/SMST_Probability_results/Probability_Based_on_Training/SVR_mean_errors_stddevs.csv")

# Load the results of the different models in a dataframe.
frames = ([NN_Rd_weigths_error,SARIMA_error,NN_error,LSTM_error, RF_error,SVR_error])
df = pd.concat(frames, axis = 0)
string = (['NN_Rd_Weights', 'SARIMA','NN','LSTM','RF','SVR'])

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
fig3, axes3 = plt.subplots(2,1,figsize=(12,10))
axes3[0].plot(x_axis, RF_mean_stddev.iloc[:,-2], label= "Mean Error", color='orange')
axes3[0].fill_between(x_axis,
                      (RF_mean_stddev.iloc[:,-2]+RF_mean_stddev.iloc[:,-1]),
                      (RF_mean_stddev.iloc[:,-2]-RF_mean_stddev.iloc[:,-1]),
                      label= "Standard Deviation" ,alpha = 0.2, color='orange')
axes3[0].set_ylabel('NN Rd Weights Error [GW]', size = 10)
axes3[0].set_xlabel("Settlement Periods", size = 14)
axes3[0].grid(True)

axes3[1].plot(x_axis, SARIMA_mean_stddev.iloc[:,-2], label= "Mean Error", color='orange')
axes3[1].fill_between(x_axis,
                      (SARIMA_mean_stddev.iloc[:,-2]+SARIMA_mean_stddev.iloc[:,-1]),
                      (SARIMA_mean_stddev.iloc[:,-2]-SARIMA_mean_stddev.iloc[:,-1]),
                      label= "Standard Deviation" ,alpha = 0.2, color='orange')
axes3[1].set_ylabel('SARIMA Error [GW]', size = 10)
axes3[1].set_xticks(np.arange(1,385, 48))
axes3[1].set_xticklabels(["", "1 / Monday", "49 / Tuesday", "97 / Wednesday", "145 / Thursday", "193 / Friday","241 / Saturday", "289 / Sunday",""])
axes3[1].set_xlabel("Settlement Period / Weekday", size = 14)
axes3[1].grid(True)
fig3.show()

# Plot the mean and standard deviations.
fig4, axes4 = plt.subplots(4,1,figsize=(12,10))
axes4[0].plot(x_axis, NN_mean_stddev.iloc[:,-2], label= "Mean Error", color='orange')
axes4[0].fill_between(x_axis,
                      (NN_mean_stddev.iloc[:,-2]+NN_mean_stddev.iloc[:,-1]),
                      (NN_mean_stddev.iloc[:,-2]-NN_mean_stddev.iloc[:,-1]),
                      label= "Standard Deviation" ,alpha = 0.2, color='orange')
axes4[0].set_ylabel('NN Error [GW]', size = 10)
axes4[0].grid(True)

axes4[1].plot(x_axis, LSTM_mean_stddev.iloc[:,-2], label= "Mean Error", color='orange')
axes4[1].fill_between(x_axis,
                      (LSTM_mean_stddev.iloc[:,-2]+LSTM_mean_stddev.iloc[:,-1]),
                      (LSTM_mean_stddev.iloc[:,-2]-LSTM_mean_stddev.iloc[:,-1]),
                      label= "Standard Deviation" ,alpha = 0.2, color='orange')
axes4[1].set_ylabel('LSTM Error [GW]', size = 10)
axes4[1].grid(True)

axes4[2].plot(x_axis, RF_mean_stddev.iloc[:,-2], label= "Mean Error", color='orange')
axes4[2].fill_between(x_axis,
                      (RF_mean_stddev.iloc[:,-2]+RF_mean_stddev.iloc[:,-1]),
                      (RF_mean_stddev.iloc[:,-2]-RF_mean_stddev.iloc[:,-1]),
                      label= "Standard Deviation" ,alpha = 0.2, color='orange')
axes4[2].set_ylabel('RF Error [GW]', size = 10)
axes4[2].grid(True)

axes4[3].plot(x_axis, SVR_mean_stddev.iloc[:,-2], label= "Mean Error", color='orange')
axes4[3].fill_between(x_axis,
                      (SVR_mean_stddev.iloc[:,-2]+SVR_mean_stddev.iloc[:,-1]),
                      (SVR_mean_stddev.iloc[:,-2]-SVR_mean_stddev.iloc[:,-1]),
                      label= "Standard Deviation" ,alpha = 0.2, color='orange')
axes4[3].set_ylabel('SVR Error [GW]', size = 10)
axes4[3].set_xticks(np.arange(1,385, 48))
axes4[3].set_xticklabels(["", "1 / Monday", "49 / Tuesday", "97 / Wednesday", "145 / Thursday", "193 / Friday","241 / Saturday", "289 / Sunday",""])
axes4[3].set_xlabel("Settlement Period / Weekday", size = 14)
axes4[3].grid(True)
fig4.show()
