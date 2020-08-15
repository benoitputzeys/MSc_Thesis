import pandas as pd
import matplotlib.pyplot as plt

# Load the results of the different models in respective variables.
Naive = pd.read_csv("Compare_Models/Direct_Multi_Step_Results/Naive.csv")
ANN = pd.read_csv("Compare_Models/Direct_Multi_Step_Results/ANN.csv")
DT = pd.read_csv("Compare_Models/Direct_Multi_Step_Results/DT.csv")
LSTM = pd.read_csv("Compare_Models/Direct_Multi_Step_Results/LSTM.csv")
Random_Forest = pd.read_csv("Compare_Models/Direct_Multi_Step_Results/RF.csv")
SVR = pd.read_csv("Compare_Models/Direct_Multi_Step_Results/SVR.csv")

# Load the results of the different models in a dataframe.
frames = ([Naive, DT, ANN, Random_Forest, SVR, LSTM])
df = pd.concat(frames, axis = 0)
string = (['DT', 'Naive', 'ANN','LSTM','RF','SVR'])

# Create histograms for RMSE, MSE and MAE of the Training Set
fig2, axes2 = plt.subplots(1,3,figsize=(12,6))
fig2.suptitle("Training Set Errors",fontsize =14)
axes2[0].bar(['Naive','DT','NN','RF','SVR','LSTM'], df.iloc[:,1], color='blue')
axes2[0].set_ylabel('MSE, GW^2', size = 14)
axes2[0].set_xticklabels(rotation=0, labels = string)
axes2[0].grid(True)

axes2[1].bar(['Naive','DT','NN','RF','SVR','LSTM'], df.iloc[:,2], color='blue')
axes2[1].set_ylabel('MAE, GW', size = 14)
axes2[1].set_xticklabels(rotation=0, labels = string)
axes2[1].grid(True)

axes2[2].bar(['Naive','DT','NN','RF','SVR','LSTM'], df.iloc[:,3], color='blue')
axes2[2].set_ylabel('RMSE, GW', size = 14)
axes2[2].grid(True)
axes2[2].set_xticklabels(rotation=0, labels = string)
fig2.subplots_adjust(top = 0.25, wspace = 200)
fig2.show()

# Create histograms for RMSE, MSE and MAE of the Test Set
fig3, axes3 = plt.subplots(1,3,figsize=(12,6))
fig3.suptitle("Test Set Errors",fontsize =14)
axes3[0].bar(['Naive','DT','NN','RF','SVR','LSTM'], df.iloc[:,4], color='blue')
axes3[0].set_ylabel('MSE, GW^2', size = 14)
axes3[0].set_xticklabels(rotation=0, labels = string)
axes3[0].grid(True)

axes3[1].bar(['Naive','DT','NN','RF','SVR','LSTM'], df.iloc[:,5], color='blue')
axes3[1].set_ylabel('MAE, GW', size = 14)
axes3[1].set_xticklabels(rotation=0, labels = string)
axes3[1].grid(True)

axes3[2].bar(['Naive','DT','NN','RF','SVR','LSTM'], df.iloc[:,6], color='blue')
axes3[2].set_ylabel('RMSE, GW', size = 14)
axes3[2].grid(True)
axes3[2].set_xticklabels(rotation=0, labels = string)
fig3.subplots_adjust(top = 0.25, wspace = 200)
fig3.show()
