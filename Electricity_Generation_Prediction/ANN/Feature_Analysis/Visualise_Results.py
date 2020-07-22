import pandas as pd
import matplotlib.pyplot as plt

########################################################################################################################
# Compare how including the dates as features has an impact on the prediction.
########################################################################################################################

# Load the results in respective variables.
F6 = pd.read_csv("Electricity_Generation_Prediction/ANN/Feature_Analysis/F6.csv")
F6_SP = pd.read_csv("Electricity_Generation_Prediction/ANN/Feature_Analysis/F6_SP.csv")
F6_SP_DoW = pd.read_csv("Electricity_Generation_Prediction/ANN/Feature_Analysis/F6_SP_DoW.csv")
F6_SP_DoW_D = pd.read_csv("Electricity_Generation_Prediction/ANN/Feature_Analysis/F6_SP_DoW_D.csv")
F6_SP_DoW_D_M = pd.read_csv("Electricity_Generation_Prediction/ANN/Feature_Analysis/F6_SP_DoW_D_M.csv")
F6_SP_DoW_D_M_Y = pd.read_csv("Electricity_Generation_Prediction/ANN/Feature_Analysis/F6_SP_DoW_D_M_Y.csv")

# Load the results in a dataframe.
frames = ([ F6_SP_DoW_D_M_Y, F6_SP_DoW_D_M,F6_SP_DoW_D, F6_SP_DoW,F6_SP, F6])
df = pd.concat(frames, axis = 0)
string = ['6_SP_DoW_D_M_Y', '6_SP_DoW_D_M','6_SP_DoW_D', '6_SP_DoW','6_SP', '6']

# Create histograms for RMSE, MSE and MAE.
fig, axes = plt.subplots(1,3,figsize=(12,6))
axes[0].bar(df.iloc[:,0], df.iloc[:,1]/1000000, color='blue')
axes[0].set_ylabel('MSE [GW^2]', size = 14)
axes[0].set_xticklabels(rotation=90, labels = string)
axes[0].grid(True)

axes[1].bar(df.iloc[:,0], df.iloc[:,2]/1000, color='blue')
axes[1].set_ylabel('MAE [GW]', size = 14)
axes[1].set_xticklabels(rotation=90, labels = string)
axes[1].grid(True)

axes[2].bar(df.iloc[:,0], df.iloc[:,3]/1000, color='blue')
axes[2].set_ylabel('RMSE [GW]', size = 14)
axes[2].grid(True)
axes[2].set_xticklabels(rotation=90, labels = string)
fig.show()

########################################################################################################################
# Compare how the 11 features (with dates) fare against 6 features (no dates).
########################################################################################################################

# Load the results in respective variables.
F11_Single_Step = pd.read_csv("Electricity_Generation_Prediction/ANN/Feature_Analysis/F11_Single_Step.csv")
F6_Single_Step = pd.read_csv("Electricity_Generation_Prediction/ANN/Feature_Analysis/F6_Single_Step.csv")

# Load the results in a dataframe.
frames = ([ F11_Single_Step, F6_Single_Step])
df = pd.concat(frames, axis = 0)
string = ['F11_Single_Step', 'F6_Single_Step']

# Create histograms for RMSE, MSE and MAE.
fig2, axes2 = plt.subplots(1,3,figsize=(12,6))
axes2[0].bar(df.iloc[:,0], df.iloc[:,1]/1000000, color='blue')
axes2[0].set_ylabel('MSE [GW^2]', size = 14)
axes2[0].set_xticklabels(rotation=90, labels = string)
axes2[0].grid(True)

axes2[1].bar(df.iloc[:,0], df.iloc[:,2]/1000, color='blue')
axes2[1].set_ylabel('MAE [GW]', size = 14)
axes2[1].set_xticklabels(rotation=90, labels = string)
axes2[1].grid(True)

axes2[2].bar(df.iloc[:,0], df.iloc[:,3]/1000, color='blue')
axes2[2].set_ylabel('RMSE [GW]', size = 14)
axes2[2].grid(True)
axes2[2].set_xticklabels(rotation=90, labels = string)
fig2.show()

