import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

X = pd.read_csv('Data_Preprocessing/For_336_SP_Step_Prediction/X.csv', delimiter=',')
load = X.iloc[-48*3-25:-25,1]/1000

# Plot
fig1, axs1=plt.subplots(1,1,figsize=(8,6))
axs1.scatter(X.iloc[-48*3-25:-25,0],load, color = "blue")
axs1.set_xlabel('Input x',size = 18)
axs1.set_ylabel('Electricity Load, GW',size = 18)

# Include additional details such as tick intervals, rotation, legend positioning and grid on.
axs1.grid(True)
axs1.set_xticks([]), axs1.set_yticks([])
axs1.set(xlim=(-10, 170), ylim=(10,37))
fig1.show()
fig1.savefig("Load_Prediction/ANN/Figures/Regression_Explained.pdf", bbox_inches='tight')

