import pandas as pd
import matplotlib.pyplot as plt

ANN = pd.read_csv("/Users/benoitputzeys/PycharmProjects/MSc_Thesis/Compare_Models/ANN_result.csv")
Decision_Tree = pd.read_csv("/Users/benoitputzeys/PycharmProjects/MSc_Thesis/Compare_Models/Decision_Tree_result.csv")
LSTM = pd.read_csv("/Users/benoitputzeys/PycharmProjects/MSc_Thesis/Compare_Models/LSTM_result.csv")
Random_Forest = pd.read_csv("/Users/benoitputzeys/PycharmProjects/MSc_Thesis/Compare_Models/Random_Forest_result.csv")
SVR = pd.read_csv("/Users/benoitputzeys/PycharmProjects/MSc_Thesis/Compare_Models/SVR_result.csv")
Previous_Day = pd.read_csv("/Users/benoitputzeys/PycharmProjects/MSc_Thesis/Compare_Models/Previous_Day_result.csv")

frames = ([ANN, Decision_Tree, LSTM, Random_Forest, SVR, Previous_Day])
df = pd.concat(frames, axis = 0)
string = (['MSE','MAE','RMSE'])

# Create bars and choose color
for i in range(1,df.shape[1]):
    plt.figure()
    plt.bar(df.iloc[:,0], df.iloc[:,i], color='blue')
    # Add title and axis names
    plt.title('Comparing Models with one another for electricity generation prediction.')
    plt.xlabel('Methods Used')
    plt.ylabel(string[i-1])

    # Show graphic
    plt.show()
