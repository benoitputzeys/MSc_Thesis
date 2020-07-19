import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

# Get the X (containing the features) and y (containing the labels) values
X = pd.read_csv('Data_Preprocessing/For_Multi_Step_Prediction/X.csv', delimiter=',')
X = X.drop(['Unnamed: 0'], axis=1)
dates = X.iloc[:,-1]
X = X.iloc[:,:-1]

y = pd.read_csv('Data_Preprocessing/For_Multi_Step_Prediction/y.csv', delimiter=',')
y = y.drop(['Unnamed: 0'], axis=1)
y = y.iloc[:,-1]

# Combine the actual load with the given features in X.
X["y"]=y

# Compute the correlation matrix.
correlation_matrix = X.corr()
print(correlation_matrix)

# Plot the correlation matrix heat map using seabor.
fig, ax = plt.subplots(figsize=(10,10))
sn.heatmap(correlation_matrix, annot=True,linewidths=.5, ax=ax)
fig.show()
