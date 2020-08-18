import pandas as pd
import matplotlib.pyplot as plt

# Load the results of the different models in respective variables.
Model_1 = pd.read_csv("Load_Prediction/ANN/Impact_of_Random_Initialization/Results/NN_error_Model_1.csv")
Model_2 = pd.read_csv("Load_Prediction/ANN/Impact_of_Random_Initialization/Results/NN_error_Model_2.csv")

# Load the results of the different models in a dataframe.
frames = ([Model_1,Model_2])
df = pd.concat(frames, axis = 0)
df.iloc[0,0] = "Method 1"
df.iloc[1,0] = "Method 2"
string = (['Model 1', 'Model 2'])

# Create histograms for RMSE, MSE and MAE.
fig2, axes2 = plt.subplots(1,3,figsize=(12,6))
axes2[0].bar(df.iloc[:,0], df.iloc[:,-3], color='blue')
axes2[0].set_ylabel('MSE, GW^2', size = 14)
axes2[0].set_xticklabels(rotation=0, labels = string)
axes2[0].grid(True)

axes2[1].bar(df.iloc[:,0], df.iloc[:,-2], color='blue')
axes2[1].set_ylabel('MAE, GW', size = 14)
axes2[1].set_xticklabels(rotation=0, labels = string)
axes2[1].grid(True)

axes2[2].bar(df.iloc[:,0], df.iloc[:,-1], color='blue')
axes2[2].set_ylabel('RMSE, GW', size = 14)
axes2[2].grid(True)
axes2[2].set_xticklabels(rotation=0, labels = string)
fig2.show()
fig2.savefig("Load_Prediction/ANN/Figures/Random_Initialization.pdf", bbox_inches='tight')
