import pandas as pd
import matplotlib.pyplot as plt

########################################################################################################################
# Compare how including the dates as features has an impact on the prediction.
########################################################################################################################

# Load the results in respective variables.
F7 = pd.read_csv("Load_Prediction/LSTM/Feature_Analysis/F7.csv")
F7_SP = pd.read_csv("Load_Prediction/LSTM/Feature_Analysis/F7_SP.csv")
F7_SP_DoW = pd.read_csv("Load_Prediction/LSTM/Feature_Analysis/F7_SP_DoW.csv")
F7_SP_DoW_D = pd.read_csv("Load_Prediction/LSTM/Feature_Analysis/F7_SP_DoW_D.csv")
F7_SP_DoW_D_M = pd.read_csv("Load_Prediction/LSTM/Feature_Analysis/F7_SP_DoW_D_M.csv")
F7_SP_DoW_D_M_Y = pd.read_csv("Load_Prediction/LSTM/Feature_Analysis/F7_SP_DoW_D_M_Y.csv")

# Load the results in a dataframe.
frames = ([ F7_SP_DoW_D_M_Y, F7_SP_DoW_D_M,F7_SP_DoW_D, F7_SP_DoW,F7_SP, F7])
df = pd.concat(frames, axis = 0)
string = ['F7_SP_DoW_D_M_Y', 'F7_SP_DoW_D_M','F7_SP_DoW_D', 'F7_SP_DoW','F7_SP', 'F7']

# Create histograms for RMSE, MSE and MAE.
fig, axes = plt.subplots(1,3,figsize=(12,6))
axes[0].bar(df.iloc[:,0], df.iloc[:,1]/1000000, color='blue')
axes[0].set_ylabel('MSE, GW^2', size = 14)
axes[0].set_xticklabels(rotation=90, labels = string)
axes[0].grid(True)

axes[1].bar(df.iloc[:,0], df.iloc[:,2]/1000, color='blue')
axes[1].set_ylabel('MAE, GW', size = 14)
axes[1].set_xticklabels(rotation=90, labels = string)
axes[1].grid(True)

axes[2].bar(df.iloc[:,0], df.iloc[:,3]/1000, color='blue')
axes[2].set_ylabel('RMSE, GW', size = 14)
axes[2].grid(True)
axes[2].set_xticklabels(rotation=90, labels = string)
fig.show()
fig.savefig("Load_Prediction/LSTM/Figures/Histograms_Impact_of_Date_Features.pdf", bbox_inches='tight')

