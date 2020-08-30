import pandas as pd
import matplotlib.pyplot as plt

########################################################################################################################
# Compare how including the transmission compares against not including it.
########################################################################################################################

# Load the results in respective variables.
F6_No_Transmission = pd.read_csv("Load_Prediction/ANN/Feature_Analysis/Transmission_vs_No_Transmission/F6_(No_Transmission).csv")
F7 = pd.read_csv("Load_Prediction/ANN/Feature_Analysis/F7.csv")

# Load the results in a dataframe.
frames = ([ F6_No_Transmission, F7])
df = pd.concat(frames, axis = 0)
string = ['F6_(No_Transmission)', 'F7']

# Create histograms for RMSE, MSE and MAE.
fig3, axes3 = plt.subplots(1,3,figsize=(12,6))
axes3[0].bar(df.iloc[:,0], df.iloc[:,1]/1000000, color='blue')
axes3[0].set_ylabel('MSE, GW^2', size = 14)
axes3[0].set_xticklabels(rotation=0, labels = string)
axes3[0].grid(True)

axes3[1].bar(df.iloc[:,0], df.iloc[:,2]/1000, color='blue')
axes3[1].set_ylabel('MAE, GW', size = 14)
axes3[1].set_xticklabels(rotation=0, labels = string)
axes3[1].grid(True)

axes3[2].bar(df.iloc[:,0], df.iloc[:,3]/1000, color='blue')
axes3[2].set_ylabel('RMSE, GW', size = 14)
axes3[2].grid(True)
axes3[2].set_xticklabels(rotation=0, labels = string)
axes3[0].set_axisbelow(True), axes3[1].set_axisbelow(True), axes3[2].set_axisbelow(True)
fig3.show()
fig3.savefig("Load_Prediction/ANN/Figures/Histograms_Transmission_vs_No_Transmission.pdf", bbox_inches='tight')


