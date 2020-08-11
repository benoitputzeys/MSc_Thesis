import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

# Get the X (containing the features) and y (containing the labels) values
X = pd.read_csv('Data_Preprocessing/For_336_SP_Step_Prediction/X.csv', delimiter=',')
X = X.set_index("Time")

y = pd.read_csv('Data_Preprocessing/For_336_SP_Step_Prediction/y.csv', delimiter=',')
y = y.set_index("Time")

X = X.iloc[:,:-6]

# Combine the actual load with the given features in X.
X["Actual Load"]=y
X = X.rename(columns={"Load_Past": "Load from\n1 Week ago",
                  "Transmission_Past": "Net Transmission\ninto GB",
                  "Simple_Moving_Average_10_SP": "SMA\n(10 SP)",
                  "Simple_Moving_Average_50_SP": "SMA\n(50 SP)",
                  "Simple_Moving_Average_336_SP": "SMA\n(336 SP)",
                  "Exp_Moving_Average_20_SP": "EMA\n(20 SP)",
                  "Exp_Moving_Average_50_SP": "EMA\n(50 SP)",
                  })


# Compute the correlation matrix.
correlation_matrix = X.corr()

# Plot the correlation matrix heat map using seabor.
fig, ax = plt.subplots(figsize=(10,10))
sn.heatmap(correlation_matrix, annot=True,linewidths=.5, ax=ax, annot_kws={"size":14}, cmap="Blues", fmt='.2f')
ax.tick_params(axis = "both", labelsize = 14),
fig.show()
fig.savefig("Data_Preprocessing/Figures/Correlation_Matrix_Chosen_Features.pdf", bbox_inches='tight')

