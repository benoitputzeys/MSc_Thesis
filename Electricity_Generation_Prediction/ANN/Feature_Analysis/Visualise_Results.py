import pandas as pd
import matplotlib.pyplot as plt

AF = pd.read_csv("Electricity_Generation_Prediction/ANN/Feature_Analysis/AF.csv")
AF_No_Dates = pd.read_csv("Electricity_Generation_Prediction/ANN/Feature_Analysis/AF_No_Dates.csv")
AF_No_Trans = pd.read_csv("Electricity_Generation_Prediction/ANN/Feature_Analysis/AF_No_Transmission.csv")
AF_SP = pd.read_csv("Electricity_Generation_Prediction/ANN/Feature_Analysis/AF_SP.csv")
AF_SP_DoW = pd.read_csv("Electricity_Generation_Prediction/ANN/Feature_Analysis/AF_SP_DoW.csv")
AF_SP_DoW_D = pd.read_csv("Electricity_Generation_Prediction/ANN/Feature_Analysis/AF_SP_DoW_D.csv")
AF_SP_DoW_D_M = pd.read_csv("Electricity_Generation_Prediction/ANN/Feature_Analysis/AF_SP_DoW_D_M.csv")
AF_SP_DoW_D_M_Y = pd.read_csv("Electricity_Generation_Prediction/ANN/Feature_Analysis/AF_SP_DoW_D_M_Y.csv")

frames = ([AF_No_Trans, AF, AF_No_Dates, AF_SP, AF_SP_DoW, AF_SP_DoW_D, AF_SP_DoW_D_M, AF_SP_DoW_D_M_Y])
df = pd.concat(frames, axis = 0)
string = (['MSE','MAE','RMSE'])

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