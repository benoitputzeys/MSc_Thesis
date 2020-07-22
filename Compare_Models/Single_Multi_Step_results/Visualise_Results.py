import pandas as pd
import matplotlib.pyplot as plt

# Load the results of the different models in respective variables.
Naive = pd.read_csv("Compare_Models/Single_Multi_Step_results/Naive.csv")
ANN = pd.read_csv("Compare_Models/Single_Multi_Step_results/ANN.csv")
DT = pd.read_csv("Compare_Models/Single_Multi_Step_results/DT.csv")
LSTM = pd.read_csv("Compare_Models/Single_Multi_Step_results/LSTM.csv")
Random_Forest = pd.read_csv("Compare_Models/Single_Multi_Step_results/RF.csv")
SVR = pd.read_csv("Compare_Models/Single_Multi_Step_results/SVR.csv")

# Load the results of the different models in a dataframe.
frames = ([DT,Naive, ANN, LSTM, Random_Forest, SVR])
df = pd.concat(frames, axis = 0)
string = (['DT', 'Naive', 'ANN','LSTM','RF','SVR'])

# Create histograms for RMSE, MSE and MAE.
fig2, axes2 = plt.subplots(1,3,figsize=(12,6))
axes2[0].bar(df.iloc[:,0], df.iloc[:,1]/1000000, color='blue')
axes2[0].set_ylabel('MSE [GW^2]', size = 14)
axes2[0].set_xticklabels(rotation=0, labels = string)
axes2[0].grid(True)

axes2[1].bar(df.iloc[:,0], df.iloc[:,2]/1000, color='blue')
axes2[1].set_ylabel('MAE [GW]', size = 14)
axes2[1].set_xticklabels(rotation=0, labels = string)
axes2[1].grid(True)

axes2[2].bar(df.iloc[:,0], df.iloc[:,3]/1000, color='blue')
axes2[2].set_ylabel('RMSE [GW]', size = 14)
axes2[2].grid(True)
axes2[2].set_xticklabels(rotation=0, labels = string)
fig2.show()
