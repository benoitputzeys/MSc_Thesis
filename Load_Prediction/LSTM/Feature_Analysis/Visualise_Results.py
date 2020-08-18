import pandas as pd
import matplotlib.pyplot as plt

F11_Single_Step = pd.read_csv("Load_Prediction/LSTM/Feature_Analysis/F11_Single_Step.csv")
F6_Single_Step = pd.read_csv("Load_Prediction/LSTM/Feature_Analysis/F7_Single_Step.csv")

frames = ([ F11_Single_Step, F6_Single_Step])
df = pd.concat(frames, axis = 0)
string = ['F11_Single_Step', 'F6_Single_Step']

# Create bars and choose color
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

