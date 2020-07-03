import pandas as pd
import matplotlib.pyplot as plt

ANN = pd.read_csv("Compare_Models/MST2_results/ANN_result.csv")
Decision_Tree = pd.read_csv("Compare_Models/MST2_results/Decision_Tree_result.csv")
LSTM = pd.read_csv("Compare_Models/MST2_results/LSTM_result.csv")
Random_Forest = pd.read_csv("Compare_Models/MST2_results/Random_Forest_result.csv")
SVR = pd.read_csv("Compare_Models/MST2_results/SVR_result.csv")
Hybrid = pd.read_csv("Compare_Models/MST2_results/Hybrid_result.csv")

frames = ([ANN, Decision_Tree, LSTM, Random_Forest, SVR, Hybrid])
#frames = ([ Decision_Tree, LSTM, Random_Forest, SVR, Previous_Day, Hybrid])
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
