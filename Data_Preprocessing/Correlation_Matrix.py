import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

# Get the X (containing the features) and y (containing the labels) values
X = pd.read_csv('Data_Preprocessing/For_336_SP_Step_Prediction/X.csv', delimiter=',')
X = X.set_index("Time")
y = pd.read_csv('Data_Preprocessing/For_336_SP_Step_Prediction/y.csv', delimiter=',')
y = y.set_index("Time")

# Get rid of unnecessary features.
X = X.iloc[:,:-6]

# Create a new dataframe that contains the input features in a ranked order from least to most important.
ordered = pd.DataFrame()
ordered["Net Transmission\ninto GB"] = X["Transmission_Past"]
ordered["SMA\n(336 SP)"] = X["Simple_Moving_Average_336_SP"]
ordered["SMA\n(48 SP)"] = X["Simple_Moving_Average_48_SP"]
ordered["EMA\n(48 SP)"] = X["Exp_Moving_Average_48_SP"]
ordered["SMA\n(10 SP)"] = X["Simple_Moving_Average_10_SP"]
ordered["EMA\n(10 SP)"] = X["Exp_Moving_Average_10_SP"]
ordered["Load from\n1 Week ago"] = X["Load_Past"]
ordered["Actual Load"]=y["Load"]

# Compute the correlation matrix.
correlation_matrix = ordered.corr()

# Plot the correlation matrix heat map using seabor.
fig, ax = plt.subplots(figsize=(10,10))
sn.heatmap(correlation_matrix, annot=True,linewidths=.5, ax=ax, annot_kws={"size":14}, cmap="Blues", fmt='.2f')
ax.tick_params(axis = "both", labelsize = 14),
fig.show()
# Save the correlation matrix.
fig.savefig("Data_Preprocessing/Figures/Correlation_Matrix.pdf", bbox_inches='tight')

