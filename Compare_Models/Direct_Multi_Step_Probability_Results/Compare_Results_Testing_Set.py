import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the results of the different models in respective variables.
# Probability based on the models
NN_Rd_weigths_error = pd.read_csv("Compare_Models/Direct_Multi_Step_Probability_Results/Probability_Based_on_Model/NN_error.csv")
NN_Rd_weigths_mean_stddev_train = pd.read_csv("Compare_Models/Direct_Multi_Step_Probability_Results/Probability_Based_on_Model/NN_mean_errors_stddevs_train.csv")
NN_Rd_weigths_mean_stddev_test = pd.read_csv("Compare_Models/Direct_Multi_Step_Probability_Results/Probability_Based_on_Model/NN_mean_errors_stddevs_test.csv")
SARIMA_error = pd.read_csv("Compare_Models/Direct_Multi_Step_Probability_Results/Probability_Based_on_Model/SARIMA_error.csv")
SARIMA_mean_stddev = pd.read_csv("Compare_Models/Direct_Multi_Step_Probability_Results/Probability_Based_on_Model/SARIMA_mean_errors_stddevs.csv")

# Errors during training and testing
NN_error = pd.read_csv("Compare_Models/Direct_Multi_Step_Probability_Results/Probability_Based_on_Training/NN_error.csv")
DT_error = pd.read_csv("Compare_Models/Direct_Multi_Step_Probability_Results/Probability_Based_on_Training/DT_error.csv")
RF_error = pd.read_csv("Compare_Models/Direct_Multi_Step_Probability_Results/Probability_Based_on_Training/RF_error.csv")
LSTM_error = pd.read_csv("Compare_Models/Direct_Multi_Step_Probability_Results/Probability_Based_on_Training/LSTM_error.csv")
SVR_error = pd.read_csv("Compare_Models/Direct_Multi_Step_Probability_Results/Probability_Based_on_Training/SVR_error.csv")
Naive_error = pd.read_csv("Compare_Models/Direct_Multi_Step_Probability_Results/Probability_Based_on_Training/Naive_error.csv")

Historic_mean_stddev_training = pd.read_csv("Compare_Models/Direct_Multi_Step_Probability_Results/Probability_Based_on_Training/Historic_mean_and_stddevs_train.csv")

# Stddevs based on the testing
NN_mean_stddev_test = pd.read_csv("Compare_Models/Direct_Multi_Step_Probability_Results/Probability_Based_on_Training/NN_mean_errors_stddevs_test.csv")
LSTM_mean_stddev_test = pd.read_csv("Compare_Models/Direct_Multi_Step_Probability_Results/Probability_Based_on_Training/LSTM_mean_errors_stddevs_test.csv")
DT_mean_stddev_test = pd.read_csv("Compare_Models/Direct_Multi_Step_Probability_Results/Probability_Based_on_Training/DT_mean_errors_stddevs_test.csv")
RF_mean_stddev_test = pd.read_csv("Compare_Models/Direct_Multi_Step_Probability_Results/Probability_Based_on_Training/RF_mean_errors_stddevs_test.csv")
SVR_mean_stddev_test = pd.read_csv("Compare_Models/Direct_Multi_Step_Probability_Results/Probability_Based_on_Training/SVR_mean_errors_stddevs_test.csv")

# Load the results of the different models in a dataframe.
training_errors = ([Naive_error.iloc[:,1:4],
                    DT_error.iloc[:,1:4],
                    NN_error.iloc[:, 1:4],
                    RF_error.iloc[:,1:4],
                    SVR_error.iloc[:, 1:4],
                    LSTM_error.iloc[:, 1:4]])
test_errors = ([Naive_error.iloc[:,-3:],
                DT_error.iloc[:,-3:],
                NN_error.iloc[:,-3:],
                RF_error.iloc[:,-3:],
                SVR_error.iloc[:, -3:],
                LSTM_error.iloc[:, -3:]])

df_training_errors = pd.concat(training_errors, axis = 0)
df_test_errors = pd.concat(test_errors, axis = 0)

########################################################################################################################
# Histograms for errors on the prediction of the training and test set
########################################################################################################################

# Create histograms for RMSE, MSE and MAE for the training set.
fig2, axes2 = plt.subplots(1,3,figsize=(12,6))
fig2.suptitle("Training Set Errors", fontsize =14)
axes2[0].bar(['Naive','DT','NN','RF','SVR','LSTM'],df_training_errors.iloc[:,0], color='blue')
axes2[0].set_ylabel('MSE, GW^2', size = 14)
axes2[0].set_xticklabels(rotation=0, labels = ['Naive','DT','NN','RF','SVR','LSTM'])
axes2[0].grid(True)

axes2[1].bar(['Naive','DT','NN','LSTM','RF','SVR'],df_training_errors.iloc[:,1], color='blue')
axes2[1].set_ylabel('MAE, GW', size = 14)
axes2[1].set_xticklabels(rotation=0, labels = ['Naive','DT','NN','RF','SVR','LSTM'])
axes2[1].grid(True)

axes2[2].bar(['Naive','DT','NN','RF','SVR','LSTM'],df_training_errors.iloc[:,2], color='blue')
axes2[2].set_ylabel('RMSE, GW', size = 14)
axes2[2].grid(True)
axes2[2].set_xticklabels(rotation=0, labels = ['Naive','DT','NN','RF','SVR','LSTM'])
fig2.subplots_adjust(top = 0.25, wspace = 200)
fig2.show()
fig2.savefig("Compare_Models/Direct_Multi_Step_Probability_Results/Figures/Training_Set_Errors.pdf", bbox_inches='tight')

# Create histograms for RMSE, MSE and MAE for the test set.
fig3, axes3 = plt.subplots(1,3,figsize=(12,6))
fig3.suptitle("Test Set Errors",fontsize =14)
axes3[0].bar(['Naive','DT','NN','RF','SVR','LSTM'],df_test_errors.iloc[:,0], color='blue')
axes3[0].set_ylabel('MSE, GW^2', size = 14)
axes3[0].set_xticklabels(rotation=0, labels = ['Naive','DT','NN','RF','SVR','LSTM'])
axes3[0].grid(True)

axes3[1].bar(['Naive','DT','NN','LSTM','RF','SVR'], df_test_errors.iloc[:,1], color='blue')
axes3[1].set_ylabel('MAE, GW', size = 14)
axes3[1].set_xticklabels(rotation=0, labels = ['Naive','DT','NN','RF','SVR','LSTM'])
axes3[1].grid(True)

axes3[2].bar(['Naive','DT','NN','RF','SVR','LSTM'],df_test_errors.iloc[:,2], color='blue')
axes3[2].set_ylabel('RMSE, GW', size = 14)
axes3[2].grid(True)
axes3[2].set_xticklabels(rotation=0, labels = ['Naive','DT','NN','RF','SVR','LSTM'])
fig3.subplots_adjust(top = 0.25, wspace = 200)
fig3.show()
fig3.savefig("Compare_Models/Direct_Multi_Step_Probability_Results/Figures/Test_Set_Errors.pdf", bbox_inches='tight')

########################################################################################################################
# Compare the mean and standard deviations of errors of the different models between their predictions and the
# true values of the training. In thus big comparison, the decision tree is not included as the RF should be the
# superior choice.
########################################################################################################################

mean_error_NN = NN_mean_stddev_test.iloc[:,0] - NN_mean_stddev_test.iloc[:,2]
stddev_NN = NN_mean_stddev_test.iloc[:,1]

mean_error_SARIMA = SARIMA_mean_stddev.iloc[:,1]
stddev_SARIMA = SARIMA_mean_stddev.iloc[:,2]/1000

x_axis = np.linspace(1,336,336)
zeros = np.zeros((336,))

# Compare the mean and standard deviations of errors of the NN between predictions and true values of the training set.
fig4, axes4 = plt.subplots(4,1,figsize=(12,10))
axes4[0].plot(x_axis, NN_mean_stddev_test.iloc[:,-2], color='orange')
axes4[0].fill_between(x_axis,
                   (NN_mean_stddev_test.iloc[:,-2]+NN_mean_stddev_test.iloc[:,-1]),
                   (NN_mean_stddev_test.iloc[:,-2]-NN_mean_stddev_test.iloc[:,-1]),
                   alpha = 0.2, color='orange')
axes4[0].set_ylim([-5.4,5.4])
axes4[1].plot(x_axis, LSTM_mean_stddev_test.iloc[:,-2], color='orange')
axes4[1].fill_between(x_axis,
                   (LSTM_mean_stddev_test.iloc[:,-2]+LSTM_mean_stddev_test.iloc[:,-1]),
                   (LSTM_mean_stddev_test.iloc[:,-2]-LSTM_mean_stddev_test.iloc[:,-1]),
                   alpha = 0.2, color='orange')
axes4[1].set_ylim([-5.4,5.4])
axes4[2].plot(x_axis, RF_mean_stddev_test.iloc[:,-2], color='orange')
axes4[2].fill_between(x_axis,
                   (RF_mean_stddev_test.iloc[:,-2]+RF_mean_stddev_test.iloc[:,-1]),
                   (RF_mean_stddev_test.iloc[:,-2]-RF_mean_stddev_test.iloc[:,-1]),
                   alpha = 0.2, color='orange')
axes4[2].set_ylim([-5.4,5.4])
axes4[3].plot(x_axis, SVR_mean_stddev_test.iloc[:,-2], color='orange')
axes4[3].fill_between(x_axis,
                   (SVR_mean_stddev_test.iloc[:,-2]+SVR_mean_stddev_test.iloc[:,-1]),
                   (SVR_mean_stddev_test.iloc[:,-2]-SVR_mean_stddev_test.iloc[:,-1]),
                   alpha = 0.2, color='orange')
axes4[3].set_ylim([-5.4,5.4])

axes4[0].set_ylabel('NN Error, GW', size = 12), axes4[1].set_ylabel('LSTM Error, GW', size = 12), axes4[2].set_ylabel('RF Error, GW', size = 12), axes4[3].set_ylabel('SVR Error, GW', size = 12)
axes4[0].set_xticks(np.arange(1,385, 24)), axes4[1].set_xticks(np.arange(1,385, 24)), axes4[2].set_xticks(np.arange(1,385, 24)), axes4[3].set_xticks(np.arange(1,385, 24))

axes4[0].set_xticklabels([]), axes4[1].set_xticklabels([]), axes4[2].set_xticklabels([])
axes4[3].set_xticklabels(["00:00\nMonday","12:00",
                       "00:00\nTuesday","12:00",
                       "00:00\nWednesday", "12:00",
                       "00:00\nThursday", "12:00",
                       "00:00\nFriday","12:00",
                       "00:00\nSaturday", "12:00",
                       "00:00\nSunday","12:00",
                       "00:00"])
axes4[0].grid(True), axes4[1].grid(True), axes4[2].grid(True), axes4[3].grid(True)
axes4[0].minorticks_on(), axes4[1].minorticks_on(), axes4[2].minorticks_on(), axes4[3].minorticks_on()
axes4[0].grid(b=True, which='major'), axes4[1].grid(b=True, which='major'), axes4[2].grid(b=True, which='major'), axes4[3].grid(b=True, which='major')
axes4[0].grid(b=True, which='minor',alpha = 0.2), axes4[1].grid(b=True, which='minor',alpha = 0.2), axes4[2].grid(b=True, which='minor',alpha = 0.2), axes4[3].grid(b=True, which='minor',alpha = 0.2)
axes4[0].tick_params(axis = "both", labelsize = 12), axes4[1].tick_params(axis = "both", labelsize = 12), axes4[2].tick_params(axis = "both", labelsize = 12), axes4[3].tick_params(axis = "both", labelsize = 11)
fig4.show()
fig4.savefig("Compare_Models/Direct_Multi_Step_Probability_Results/Figures/Test_Set/All_Stddevs_of_Errors_Test.pdf", bbox_inches='tight')

########################################################################################################################
# Compare the mean and standard deviations of errors of the NN between predictions and true values of the test set.
########################################################################################################################

fig5, axes5 = plt.subplots(1,1,figsize=(12,6))
axes5.fill_between(x_axis,
                   (+NN_mean_stddev_test.iloc[:,-1]),
                   (zeros),
                   label= "S. dev. of the errors of the NN prediction on the test set",
                   alpha = 0.2, color='orange')
axes5.fill_between(x_axis,
                  (+Historic_mean_stddev_training.iloc[:,-1]),
                  (zeros),
                  label= "S. dev. in the training set", alpha=0.2, color = "blue")
axes5.plot(x_axis,abs(NN_mean_stddev_test.iloc[:,-2]),label= "Error in the test set", color = "orange")
axes5.set_ylabel('Standard deviation, electricity load, GW', size = 14)
axes5.set_xticks(np.arange(1,385, 24))
axes5.set_xticklabels(["00:00\nMonday","12:00",
                       "00:00\nTuesday","12:00",
                       "00:00\nWednesday", "12:00",
                       "00:00\nThursday", "12:00",
                       "00:00\nFriday","12:00",
                       "00:00\nSaturday", "12:00",
                       "00:00\nSunday","12:00",
                       "00:00"])
axes5.grid(True)
axes5.minorticks_on()
axes5.grid(b=True, which='major')
axes5.grid(b=True, which='minor',alpha = 0.2)
axes5.legend(fontsize=14)
axes5.tick_params(axis = "both", labelsize = 11)
axes5.set_ylim([0,5.4])
fig5.show()
fig5.savefig("Compare_Models/Direct_Multi_Step_Probability_Results/Figures/Test_Set/NN_Stddev_of_Error_Test.pdf", bbox_inches='tight')

########################################################################################################################
# Compare the mean and standard deviations of errors of the LSTM between predictions and true values of the test set.
########################################################################################################################

fig6, axes6 = plt.subplots(1,1,figsize=(12,6))
axes6.fill_between(x_axis,
                   (+LSTM_mean_stddev_test.iloc[:,-1]),
                   (zeros),
                   label= "S. dev. of the errors of the LSTM prediction on the test set",
                   alpha = 0.2, color='orange')
axes6.fill_between(x_axis,
                  (+Historic_mean_stddev_training.iloc[:,-1]),
                  (zeros),
                  label= "S. Dev. in the training Set", alpha=0.2, color = "blue")
axes6.plot(x_axis,abs(LSTM_mean_stddev_test.iloc[:,-2]),label= "Error in the test set", color = "orange")
axes6.set_ylabel('Standard deviation, electricity load, GW', size = 14)
axes6.set_xticks(np.arange(1,385, 24))
axes6.set_xticklabels(["00:00\nMonday","12:00",
                       "00:00\nTuesday","12:00",
                       "00:00\nWednesday", "12:00",
                       "00:00\nThursday", "12:00",
                       "00:00\nFriday","12:00",
                       "00:00\nSaturday", "12:00",
                       "00:00\nSunday","12:00",
                       "00:00"])
axes6.grid(True)
axes6.minorticks_on()
axes6.grid(b=True, which='major')
axes6.grid(b=True, which='minor',alpha = 0.2)
axes6.legend(fontsize=14)
axes6.tick_params(axis = "both", labelsize = 11)
axes6.set_ylim([0,5.4])
fig6.show()
fig6.savefig("Compare_Models/Direct_Multi_Step_Probability_Results/Figures/Test_Set/LSTM_Stddev_of_Error_Test.pdf", bbox_inches='tight')

########################################################################################################################
# Compare the mean and standard deviations of errors of the DT between predictions and true values of the test set.
########################################################################################################################

fig77, axes77 = plt.subplots(1,1,figsize=(12,6))
# axes7.plot(x_axis, RF_mean_stddev.iloc[:,-2],
#            label= "Mean Error of the \nRF prediction on \nthe Training Set\n", color='orange')
axes77.fill_between(x_axis,
                   (+DT_mean_stddev_test.iloc[:,-1]),
                   (zeros),
                   label= "S. dev. of the errors of the DT prediction on the test set",
                   alpha = 0.2, color='orange')
axes77.fill_between(x_axis,
                  (+Historic_mean_stddev_training.iloc[:,-1]),
                   (zeros),
                   label= "S. dev. in the training set", alpha=0.2, color = "blue")
axes77.plot(x_axis,abs(DT_mean_stddev_test.iloc[:,-2]),label= "Error in the test set", color = "orange")
axes77.set_ylabel('Standard deviation, electricity load, GW', size = 14)
axes77.set_xticks(np.arange(1,385, 24))
axes77.set_xticklabels(["00:00\nMonday","12:00",
                       "00:00\nTuesday","12:00",
                       "00:00\nWednesday", "12:00",
                       "00:00\nThursday", "12:00",
                       "00:00\nFriday","12:00",
                       "00:00\nSaturday", "12:00",
                       "00:00\nSunday","12:00",
                       "00:00"])
axes77.grid(True)
axes77.minorticks_on()
axes77.grid(b=True, which='major')
axes77.grid(b=True, which='minor',alpha = 0.2)
axes77.legend(fontsize=14)
axes77.tick_params(axis = "both", labelsize = 11)
axes77.set_ylim([0,5.4])
fig77.show()
fig77.savefig("Compare_Models/Direct_Multi_Step_Probability_Results/Figures/Test_Set/DT_Stddev_of_Error_Test.pdf", bbox_inches='tight')

########################################################################################################################
# Compare the mean and standard deviations of errors of the RF between predictions and true values of the test set.
########################################################################################################################

# Compare the mean and standard deviations of errors of the RF between predictions and true values of the test set.
fig7, axes7 = plt.subplots(1,1,figsize=(12,6))
# axes7.plot(x_axis, RF_mean_stddev.iloc[:,-2],
#            label= "Mean Error of the \nRF prediction on \nthe Training Set\n", color='orange')
axes7.fill_between(x_axis,
                   (+RF_mean_stddev_test.iloc[:,-1]),
                   (zeros),
                   label= "S. dev. of the errors of the RF prediction on the test set",
                   alpha = 0.2, color='orange')
axes7.fill_between(x_axis,
                  (+Historic_mean_stddev_training.iloc[:,-1]),
                   (zeros),
                   label= "S. dev. in the training set", alpha=0.2, color = "blue")
axes7.plot(x_axis,abs(RF_mean_stddev_test.iloc[:,-2]),label= "Error in the test set", color = "orange")
axes7.set_ylabel('Standard deviation, electricity load, GW', size = 14)
axes7.set_xticks(np.arange(1,385, 24))
axes7.set_xticklabels(["00:00\nMonday","12:00",
                       "00:00\nTuesday","12:00",
                       "00:00\nWednesday", "12:00",
                       "00:00\nThursday", "12:00",
                       "00:00\nFriday","12:00",
                       "00:00\nSaturday", "12:00",
                       "00:00\nSunday","12:00",
                       "00:00"])
axes7.grid(True)
axes7.minorticks_on()
axes7.grid(b=True, which='major')
axes7.grid(b=True, which='minor',alpha = 0.2)
axes7.legend(fontsize=14)
axes7.tick_params(axis = "both", labelsize = 11)
axes7.set_ylim([0,5.4])
fig7.show()
fig7.savefig("Compare_Models/Direct_Multi_Step_Probability_Results/Figures/Test_Set/RF_Stddev_of_Error_Test.pdf", bbox_inches='tight')

########################################################################################################################
# Compare the mean and standard deviations of errors of the SVR between predictions and true values of the test set.
########################################################################################################################

# Compare the mean and standard deviations of errors of the SVR between predictions and true values of the training set.
fig8, axes8 = plt.subplots(1,1,figsize=(12,6))
# axes8.plot(x_axis, SVR_mean_stddev.iloc[:,-2],
#            label= "Mean Error of the \nSVR prediction on \nthe Training Set\n", color='orange')
axes8.fill_between(x_axis,
                   (+SVR_mean_stddev_test.iloc[:,-1]),
                   (zeros),
                   label= "S. dev. of the errors of the SVR prediction on the test set",
                   alpha = 0.2, color='orange')
axes8.fill_between(x_axis,
                  (+Historic_mean_stddev_training.iloc[:,-1]),
                   (zeros),
                   label= "S. dev. in the training set", alpha=0.2, color = "blue")
axes8.plot(x_axis,abs(SVR_mean_stddev_test.iloc[:,-2]),label= "Error in the training set", color = "orange")
axes8.set_ylabel('Standard deviation, electricity load, GW', size = 14)
axes8.set_xticks(np.arange(1,385, 24))
axes8.set_xticklabels(["00:00\nMonday","12:00",
                       "00:00\nTuesday","12:00",
                       "00:00\nWednesday", "12:00",
                       "00:00\nThursday", "12:00",
                       "00:00\nFriday","12:00",
                       "00:00\nSaturday", "12:00",
                       "00:00\nSunday","12:00",
                       "00:00"])
axes8.grid(True)
axes8.minorticks_on()
axes8.grid(b=True, which='major')
axes8.grid(b=True, which='minor',alpha = 0.2)
axes8.legend(fontsize=14)
axes8.tick_params(axis = "both", labelsize = 11)
axes8.set_ylim([0,5.4])
fig8.show()
fig8.savefig("Compare_Models/Direct_Multi_Step_Probability_Results/Figures/Test_Set/SVR_Stddev_of_Error_Test.pdf", bbox_inches='tight')

########################################################################################################################
# Compare the mean and standard deviations of errors of the NN with Random Weigths between predictions and true values of the test set.
########################################################################################################################

# Compare the mean and standard deviations of errors of the SVR between predictions and true values of the training set.
fig10, axes10 = plt.subplots(1,1,figsize=(12,6))
# axes9.plot(x_axis, SVR_mean_stddev.iloc[:,-2],
#            label= "Mean Error of the \nSVR prediction on \nthe Training Set\n", color='orange')
axes10.fill_between(x_axis,
                   (+NN_Rd_weigths_mean_stddev_test.iloc[:,-1]),
                   (zeros),
                   label= "S. dev. of the errors of the NN (rd. weights) prediction on the test set",
                   alpha = 0.2, color='orange')
axes10.fill_between(x_axis,
                  (+Historic_mean_stddev_training.iloc[:,-1]),
                    (zeros),
                    label= "S. dev. in the training set", alpha=0.2, color = "blue")
axes10.plot(x_axis,abs(NN_Rd_weigths_mean_stddev_test.iloc[:,-2]),label= "Error in the test set", color = "orange")
axes10.set_ylabel('Standard deviation, electricity load, GW', size = 14)
axes10.set_xticks(np.arange(1,385, 24))
axes10.set_xticklabels(["00:00\nMonday","12:00",
                       "00:00\nTuesday","12:00",
                       "00:00\nWednesday", "12:00",
                       "00:00\nThursday", "12:00",
                       "00:00\nFriday","12:00",
                       "00:00\nSaturday", "12:00",
                       "00:00\nSunday","12:00",
                       "00:00"])
axes10.grid(True)
axes10.minorticks_on()
axes10.grid(b=True, which='major')
axes10.grid(b=True, which='minor',alpha = 0.2)
axes10.legend(fontsize=14)
axes10.tick_params(axis = "both", labelsize = 11)
axes10.set_ylim([0,5.4])
fig10.show()
fig10.savefig("Compare_Models/Direct_Multi_Step_Probability_Results/Figures/Test_Set/NN_Rd_Weights_Stddev_of_Error_Test.pdf", bbox_inches='tight')
