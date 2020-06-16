import pandas as pd
import matplotlib as plt

ANN = pd.read_csv("/Users/benoitputzeys/PycharmProjects/NN-Predicitons/Compare_Models/ANN_result.csv")
Decision_Tree = pd.read_csv("/Users/benoitputzeys/PycharmProjects/NN-Predicitons/Compare_Models/Decision_Tree_result.csv")
LSTM = pd.read_csv("/Users/benoitputzeys/PycharmProjects/NN-Predicitons/Compare_Models/LSTM_result.csv")
Random_Forest = pd.read_csv("/Users/benoitputzeys/PycharmProjects/NN-Predicitons/Compare_Models/Random_Forest_result.csv")
SVR = pd.read_csv("/Users/benoitputzeys/PycharmProjects/NN-Predicitons/Compare_Models/SVR_result.csv")

frames = ([ Decision_Tree, LSTM, Random_Forest, SVR])
df = pd.concat(frames, axis = 0)
df.index.name = "Method"
#df.column.name = "Mean Squarred Error"

df.plot.bar(x="Method", y="MSE", rot=70, title="Comparing Models with one another for electricity generation prediction.");
plt.show(block=True);




data = {"City": ["London", "Paris", "Rome"],

        "Visits": [20.42, 17.95, 9.7]

        };

# Dictionary loaded into a DataFrame

dataFrame = pd.DataFrame(data=data);

# Draw a vertical bar chart

dataFrame.plot.bar(x="City", y="Visits", rot=70, title="Number of tourist visits - Year 2018");

plt.show(block=True);