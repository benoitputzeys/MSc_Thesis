import pandas as pd
import matplotlib.pyplot as plt

AF_ML = pd.read_csv("Electricity_Generation_Prediction/ANN/Training_Set_Size_Analysis/AF_1L.csv")
AF_HL = pd.read_csv("Electricity_Generation_Prediction/ANN/Training_Set_Size_Analysis/AF_12L.csv")
AF_Quarter_L = pd.read_csv("Electricity_Generation_Prediction/ANN/Training_Set_Size_Analysis/AF_14L.csv")
AF_35_L = pd.read_csv("Electricity_Generation_Prediction/ANN/Training_Set_Size_Analysis/AF_35L.csv")
AF_34_L = pd.read_csv("Electricity_Generation_Prediction/ANN/Training_Set_Size_Analysis/AF_34L.csv")
AF_25_L = pd.read_csv("Electricity_Generation_Prediction/ANN/Training_Set_Size_Analysis/AF_25L.csv")

frames = ([AF_ML,AF_34_L,AF_25_L, AF_HL, AF_35_L, AF_Quarter_L])
df = pd.concat(frames, axis = 0)

# Create bars and choose color
fig, axes = plt.subplots(1,3,figsize=(12,6))
axes[0].bar(df.iloc[:,0], df.iloc[:,1]/1000000, color='blue')
axes[0].set_ylabel('MSE [GW^2]', size = 14)
axes[0].grid(True)

axes[1].bar(df.iloc[:,0], df.iloc[:,2]/1000, color='blue')
axes[1].set_ylabel('MAE [GW]', size = 14)
axes[1].grid(True)

axes[2].bar(df.iloc[:,0], df.iloc[:,3]/1000, color='blue')
axes[2].set_ylabel('RMSE [GW]', size = 14)
axes[2].grid(True)
fig.show()