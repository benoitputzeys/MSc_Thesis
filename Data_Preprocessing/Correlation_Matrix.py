import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

# Get the X (containing the features) and y (containing the labels) values
X = pd.read_csv('Data_Preprocessing/For_336_SP_Step_Prediction/X.csv', delimiter=',')
X = X.set_index("Time")

y = pd.read_csv('Data_Preprocessing/For_336_SP_Step_Prediction/y.csv', delimiter=',')
y = y.set_index("Time")

# Combine the actual load with the given features in X.
X["y"]=y

# Compute the correlation matrix.
correlation_matrix = X.corr()

# Plot the correlation matrix heat map using seabor.
fig, ax = plt.subplots(figsize=(10,10))
sn.heatmap(correlation_matrix, annot=True,linewidths=.5, ax=ax)
fig.show()
fig.savefig("Data_Preprocessing/Figures/Correlation_Matrix.pdf", bbox_inches='tight')

