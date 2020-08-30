import pandas as pd
import matplotlib.pyplot as plt

ANN = pd.read_csv("Compare_Models/Single_Step_Results/ANN_result.csv")
Decision_Tree = pd.read_csv("Compare_Models/Single_Step_Results/DT_result.csv")
LSTM = pd.read_csv("Compare_Models/Single_Step_Results/LSTM_result.csv")
Random_Forest = pd.read_csv("Compare_Models/Single_Step_Results/RF_result.csv")
SVR = pd.read_csv("Compare_Models/Single_Step_Results/SVR_result.csv")
Previous_Day = pd.read_csv("Compare_Models/Single_Step_Results/Previous_SP_result.csv")

frames = ([Previous_Day,LSTM, ANN, Decision_Tree,  Random_Forest, SVR ])
df = pd.concat(frames, axis = 0)
string = ['Previous SP', 'LSTM', 'ANN', 'DT','RF', 'SVR',]

# Create histograms for RMSE, MSE and MAE of the Test Set
fig, axes = plt.subplots(1,3,figsize=(12,6))
fig.suptitle("Test Set Errors",fontsize =14)
axes[0].bar(string, df.iloc[:,4], color='blue')
axes[0].set_ylabel('MSE, GW^2', size = 14)
axes[0].set_xticklabels(rotation=0, labels = string)
axes[0].grid(True)
axes[1].bar(string, df.iloc[:,5], color='blue')
axes[1].set_ylabel('MAE, GW', size = 14)
axes[1].set_xticklabels(rotation=0, labels = string)
axes[1].grid(True)
axes[2].bar(string, df.iloc[:,6], color='blue')
axes[2].set_ylabel('RMSE, GW', size = 14)
axes[2].grid(True)
axes[2].set_xticklabels(rotation=0, labels = string)
fig.subplots_adjust(top = 0.25, wspace = 200)
fig.show()

# Create histograms for RMSE, MSE and MAE of the Training Set
fig2, axes2 = plt.subplots(1,3,figsize=(12,6))
fig2.suptitle("Training Set Errors",fontsize =14)
axes2[0].bar(string, df.iloc[:,1], color='blue')
axes2[0].set_ylabel('MSE, GW^2', size = 14)
axes2[0].set_xticklabels(rotation=0, labels = string)
axes2[0].grid(True)
axes2[1].bar(string, df.iloc[:,2], color='blue')
axes2[1].set_ylabel('MAE, GW', size = 14)
axes2[1].set_xticklabels(rotation=0, labels = string)
axes2[1].grid(True)
axes2[2].bar(string, df.iloc[:,3], color='blue')
axes2[2].set_ylabel('RMSE, GW', size = 14)
axes2[2].grid(True)
axes2[2].set_xticklabels(rotation=0, labels = string)
fig2.subplots_adjust(top = 0.25, wspace = 200)
fig2.show()
