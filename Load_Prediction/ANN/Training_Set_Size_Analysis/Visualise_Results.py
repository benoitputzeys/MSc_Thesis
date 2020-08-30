import pandas as pd
import matplotlib.pyplot as plt

# Load the results in respective variables. 12 mean 1/2 length, 14 is 1/4 length etc.
AF_1_L = pd.read_csv("Load_Prediction/ANN/Training_Set_Size_Analysis/AF_1L.csv")
AF_12_L = pd.read_csv("Load_Prediction/ANN/Training_Set_Size_Analysis/AF_12L.csv")
AF_14_L = pd.read_csv("Load_Prediction/ANN/Training_Set_Size_Analysis/AF_14L.csv")
AF_35_L = pd.read_csv("Load_Prediction/ANN/Training_Set_Size_Analysis/AF_35L.csv")
AF_34_L = pd.read_csv("Load_Prediction/ANN/Training_Set_Size_Analysis/AF_34L.csv")
AF_25_L = pd.read_csv("Load_Prediction/ANN/Training_Set_Size_Analysis/AF_25L.csv")

# Load the results in a dataframe.
frames = ([AF_14_L, AF_25_L,  AF_12_L,AF_35_L, AF_34_L, AF_1_L])
df = pd.concat(frames, axis = 0)
df.iloc[:,0] = ['1/4 L','2/5 L','1/2 L','3/5 L','3/4 L','1 L']

# Create bars and choose color
fig, axes = plt.subplots(1,3,figsize=(12,6))
axes[0].bar(df.iloc[:,0], df.iloc[:,1]/1000000, color='blue')
axes[0].set_ylabel('MSE, GW^2', size = 14)
axes[0].grid(True)

axes[1].bar(df.iloc[:,0], df.iloc[:,2]/1000, color='blue')
axes[1].set_ylabel('MAE, GW', size = 14)
axes[1].grid(True)

axes[2].bar(df.iloc[:,0], df.iloc[:,3]/1000, color='blue')
axes[2].set_ylabel('RMSE, GW', size = 14)
axes[2].grid(True)
axes[0].set_axisbelow(True), axes[1].set_axisbelow(True), axes[2].set_axisbelow(True)
fig.show()
fig.savefig("Load_Prediction/ANN/Figures/Histograms_Analysis_of_Length_of_Training_Set.pdf", bbox_inches='tight')
