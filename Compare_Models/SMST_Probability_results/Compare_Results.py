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
Training_mean_stddev = pd.read_csv("Compare_Models/SMST_Probability_results/Probability_Based_on_Training/Training_mean_errors_stddevs.csv")

# Load the results of the different models in a dataframe.
frames = ([SARIMA_error, NN_Rd_weigths_error, NN_error,LSTM_error, RF_error,SVR_error])
df = pd.concat(frames, axis = 0)
string = ['SARIMA' ,'NN_Rd_weigths_error', 'NN','LSTM','RF','SVR']

# Create histograms for RMSE, MSE and MAE.
fig2, axes2 = plt.subplots(1,3,figsize=(12,6))
axes2[0].bar(['SARIMA' ,'NN_Rd_weigths_error', 'NN','LSTM','RF','SVR'], df.iloc[:,1], color='blue')
axes2[0].set_ylabel('MSE [GW^2]', size = 14)
axes2[0].set_xticklabels(rotation=90, labels = string)
axes2[0].grid(True)

axes2[1].bar(['SARIMA' ,'NN_Rd_weigths_error', 'NN','LSTM','RF','SVR'], df.iloc[:,2], color='blue')
axes2[1].set_ylabel('MAE [GW]', size = 14)
axes2[1].set_xticklabels(rotation=90, labels = string)
axes2[1].grid(True)

axes2[2].bar(df.iloc[0:6,0], df.iloc[0:6,3], color='blue')
axes2[2].set_ylabel('RMSE [GW]', size = 14)
axes2[2].grid(True)
axes2[2].set_xticklabels(rotation=90, labels = string)
fig2.show()

mean_error_NN = NN_mean_stddev.iloc[:,0] - NN_mean_stddev.iloc[:,2]
stddev_NN = NN_mean_stddev.iloc[:,1]

mean_error_SARIMA = SARIMA_mean_stddev.iloc[:,1]
stddev_SARIMA = SARIMA_mean_stddev.iloc[:,2]/1000

x_axis = np.linspace(1,336,336)
#
# # Compare the mean and standard deviations between SARIMA and historic values.
# fig3, axes3 = plt.subplots(1,1,figsize=(12,6))
# axes3.plot(x_axis, SARIMA_mean_stddev.iloc[:,-2],
#            label= "Mean Error of the \nSARIMA prediction \non the Training Set\n", color='orange')
# axes3.fill_between(x_axis,
#                       (SARIMA_mean_stddev.iloc[:,-2]+SARIMA_mean_stddev.iloc[:,-1]),
#                       (SARIMA_mean_stddev.iloc[:,-2]-SARIMA_mean_stddev.iloc[:,-1]),
#                       label= "Standard Deviation of\nthe SARIMA prediction \non the Training Set\n" ,alpha = 0.2, color='orange')
# axes3.fill_between(x_axis,
#                   (-Training_mean_stddev.iloc[:,-1]),
#                   (+Training_mean_stddev.iloc[:,-1]),
#                   label= "Variation in the \nTraining Set", alpha=0.2, color = "blue")
# axes3.set_ylabel('Electricity Load [GW]', size = 14)
# axes3.set_xlabel("Settlement Periods", size = 14)
# axes3.set_xticks(np.arange(1,385, 24))
# axes3.set_xticklabels(["00:00\nMonday","12:00",
#                        "00:00\nTuesday","12:00",
#                        "00:00\nWednesday", "12:00",
#                        "00:00\nThursday", "12:00",
#                        "00:00\nFriday","12:00",
#                        "00:00\nSaturday", "12:00",
#                        "00:00\nSunday","12:00",
#                        "00:00"])
# axes3.set_xlabel("Hour / Weekday", size = 14)
# axes3.minorticks_on()
# axes3.grid(b=True, which='major')
# axes3.grid(b=True, which='minor',alpha = 0.2)
# axes3.grid(True)
# axes3.legend(loc=(1.01,0.625))
# axes3.tick_params(axis = "both", labelsize = 11)
# fig3.show()
#
# # Compare the mean and standard deviations between NN Rd. Weigths and historic values.
# fig4, axes4 = plt.subplots(1,1,figsize=(12,6))
# axes4.plot(x_axis, NN_Rd_weigths_mean_stddev.iloc[:,-2],
#            label= "Mean Error of the \nNN with random \nweights prediciton\non the Training Set\n", color='orange')
# axes4.fill_between(x_axis,
#                       (+NN_Rd_weigths_mean_stddev.iloc[:,-1]),
#                       (-NN_Rd_weigths_mean_stddev.iloc[:,-1]),
#                       label= "Standard Deviation of\nthe NN with random\nweights on the\nTraining Set\n" ,alpha = 0.2, color='orange')
# axes4.fill_between(x_axis,
#                   (-Training_mean_stddev.iloc[:,-1]),
#                   (+Training_mean_stddev.iloc[:,-1]),
#                   label= "Variation in the \nTraining Set", alpha=0.2, color = "blue")
# axes4.set_ylabel('Electricity Load [GW]', size = 14)
# axes4.set_xticks(np.arange(1,385, 24))
# axes4.set_xticklabels(["00:00\nMonday","12:00",
#                        "00:00\nTuesday","12:00",
#                        "00:00\nWednesday", "12:00",
#                        "00:00\nThursday", "12:00",
#                        "00:00\nFriday","12:00",
#                        "00:00\nSaturday", "12:00",
#                        "00:00\nSunday","12:00",
#                        "00:00"])
# axes4.set_xlabel("Hour / Weekday", size = 14)
# axes4.grid(True)
# axes4.minorticks_on()
# axes4.grid(b=True, which='major')
# axes4.grid(b=True, which='minor',alpha = 0.2)
# axes4.legend(loc=(1.01,0.575))
# axes4.tick_params(axis = "both", labelsize = 11)
# fig4.show()

# Compare the mean and standard deviations between NN and historic values.
fig5, axes5 = plt.subplots(1,1,figsize=(12,6))
axes5.plot(x_axis, NN_mean_stddev.iloc[:,-2],
           label= "Mean Error of the\nNN prediction on\nthe Training Set\n", color='orange')
axes5.fill_between(x_axis,
                      (+NN_mean_stddev.iloc[:,-1]),
                      (NN_mean_stddev.iloc[:,-2]-NN_mean_stddev.iloc[:,-1]),
                      label= "Standard Deviation \nof the NN prediction \non the Training Set\n" ,alpha = 0.2, color='orange')
axes5.fill_between(x_axis,
                  (-Training_mean_stddev.iloc[:,-1]),
                  (+Training_mean_stddev.iloc[:,-1]),
                  label= "Variation in the \nTraining Set", alpha=0.2, color = "blue")
axes5.set_ylabel('Electricity Load [GW]', size = 14)
axes5.set_xticks(np.arange(1,385, 24))
axes5.set_xticklabels(["00:00\nMonday","12:00",
                       "00:00\nTuesday","12:00",
                       "00:00\nWednesday", "12:00",
                       "00:00\nThursday", "12:00",
                       "00:00\nFriday","12:00",
                       "00:00\nSaturday", "12:00",
                       "00:00\nSunday","12:00",
                       "00:00"])
axes5.set_xlabel("Hour / Weekday", size = 14)
axes5.grid(True)
axes5.minorticks_on()
axes5.grid(b=True, which='major')
axes5.grid(b=True, which='minor',alpha = 0.2)
axes5.legend(loc=(1.01,0.635))
axes5.tick_params(axis = "both", labelsize = 11)
fig5.show()

# Compare the mean and standard deviations between LSTM and historic values.
fig6, axes6 = plt.subplots(1,1,figsize=(12,6))
axes6.plot(x_axis, LSTM_mean_stddev.iloc[:,-2],
           label= "Mean Error of the LSTM prediction on the Training Set", color='orange')
axes6.fill_between(x_axis,
                      (+LSTM_mean_stddev.iloc[:,-1]),
                      (-LSTM_mean_stddev.iloc[:,-1]),
                      label= "Standard Deviation of the LSTM prediction on the Training Set" ,alpha = 0.2, color='orange')
# axes6.fill_between(x_axis,
#                   (-Training_mean_stddev.iloc[:,-1]),
#                   (+Training_mean_stddev.iloc[:,-1]),
#                   label= "Variation in the \nTraining Set", alpha=0.2, color = "blue")
axes6.set_ylabel('Electricity Load [GW]', size = 14)
axes6.set_xticks(np.arange(1,385, 24))
axes6.set_xticklabels(["00:00\nMonday","12:00",
                       "00:00\nTuesday","12:00",
                       "00:00\nWednesday", "12:00",
                       "00:00\nThursday", "12:00",
                       "00:00\nFriday","12:00",
                       "00:00\nSaturday", "12:00",
                       "00:00\nSunday","12:00",
                       "00:00"])
axes6.set_xlabel("Hour / Weekday", size = 14)
axes6.grid(True)
axes6.minorticks_on()
axes6.grid(b=True, which='major')
axes6.grid(b=True, which='minor',alpha = 0.2)
axes6.legend(fontsize=14)
axes6.tick_params(axis = "both", labelsize = 11)
fig6.show()

# Compare the mean and standard deviations between RF and historic values.
fig7, axes7 = plt.subplots(1,1,figsize=(12,6))
axes7.plot(x_axis, RF_mean_stddev.iloc[:,-2],
           label= "Mean Error of the \nRF prediction on \nthe Training Set\n", color='orange')
axes7.fill_between(x_axis,
                      (+RF_mean_stddev.iloc[:,-1]),
                      (RF_mean_stddev.iloc[:,-2]-RF_mean_stddev.iloc[:,-1]),
                      label= "Standard Deviation of\nthe RF prediction on\nthe Training Set\n" ,alpha = 0.2, color='orange')
axes7.fill_between(x_axis,
                  (-Training_mean_stddev.iloc[:,-1]),
                  (+Training_mean_stddev.iloc[:,-1]),
                  label= "Variation in the \nTraining Set", alpha=0.2, color = "blue")
axes7.set_ylabel('Electricity Load [GW]', size = 14)
axes7.set_xticks(np.arange(1,385, 24))
axes7.set_xticklabels(["00:00\nMonday","12:00",
                       "00:00\nTuesday","12:00",
                       "00:00\nWednesday", "12:00",
                       "00:00\nThursday", "12:00",
                       "00:00\nFriday","12:00",
                       "00:00\nSaturday", "12:00",
                       "00:00\nSunday","12:00",
                       "00:00"])
axes7.set_xlabel("Hour / Weekday", size = 14)
axes7.grid(True)
axes7.minorticks_on()
axes7.grid(b=True, which='major')
axes7.grid(b=True, which='minor',alpha = 0.2)
axes7.legend(loc=(1.01,0.635))
axes7.tick_params(axis = "both", labelsize = 11)
fig7.show()

# Compare the mean and standard deviations between RF and historic values.
fig8, axes8 = plt.subplots(1,1,figsize=(12,6))
axes8.plot(x_axis, SVR_mean_stddev.iloc[:,-2],
           label= "Mean Error of the \nSVR prediction on \nthe Training Set\n", color='orange')
axes8.fill_between(x_axis,
                      (+SVR_mean_stddev.iloc[:,-1]),
                      (SVR_mean_stddev.iloc[:,-2]-SVR_mean_stddev.iloc[:,-1]),
                      label= "Standard Deviation of\nthe SVR prediction on \nthe Training Set\n" ,alpha = 0.2, color='orange')
axes8.fill_between(x_axis,
                  (-Training_mean_stddev.iloc[:,-1]),
                  (+Training_mean_stddev.iloc[:,-1]),
                  label= "Variation in the \nTraining Set", alpha=0.2, color = "blue")
axes8.set_ylabel('Electricity Load [GW]', size = 14)
axes8.set_xticks(np.arange(1,385, 24))
axes8.set_xticklabels(["00:00\nMonday","12:00",
                       "00:00\nTuesday","12:00",
                       "00:00\nWednesday", "12:00",
                       "00:00\nThursday", "12:00",
                       "00:00\nFriday","12:00",
                       "00:00\nSaturday", "12:00",
                       "00:00\nSunday","12:00",
                       "00:00"])
axes8.set_xlabel("Hour / Weekday", size = 14)
axes8.grid(True)
axes8.minorticks_on()
axes8.grid(b=True, which='major')
axes8.grid(b=True, which='minor',alpha = 0.2)
axes8.legend(loc=(1.01,0.635))
axes8.tick_params(axis = "both", labelsize = 11)
fig8.show()