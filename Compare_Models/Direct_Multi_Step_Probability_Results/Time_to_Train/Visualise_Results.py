import pandas as pd
import matplotlib.pyplot as plt

# # Load the results of the different models in respective variables.
# Naive = pd.read_csv("Compare_Models/Direct_Multi_Step_Results/Naive.csv")
ANN = pd.read_csv("Compare_Models/Direct_Multi_Step_Probability_Results/Time_to_Train/ANN.csv")
NN_Rd_Weights = pd.read_csv("Compare_Models/Direct_Multi_Step_Probability_Results/Time_to_Train/NN_Rd_Weights.csv")
DT = pd.read_csv("Compare_Models/Direct_Multi_Step_Probability_Results/Time_to_Train/DT.csv")
LSTM = pd.read_csv("Compare_Models/Direct_Multi_Step_Probability_Results/Time_to_Train/LSTM.csv")
Random_Forest = pd.read_csv("Compare_Models/Direct_Multi_Step_Probability_Results/Time_to_Train/RF.csv")
SVR = pd.read_csv("Compare_Models/Direct_Multi_Step_Probability_Results/Time_to_Train/SVR.csv")

# Load the results of the different models in a dataframe.
frames = ([ DT.iloc[0,1], Random_Forest.iloc[0,1], SVR.iloc[0,1], ANN.iloc[0,1],LSTM.iloc[0,1], NN_Rd_Weights.iloc[0,1]])
string = ([ 'DT', 'RF', 'SVR','ANN','LSTM', 'ANN\nRandom Weigths'])

# Create histograms for the time elapsed when training the models.
fig2, axes2 = plt.subplots(1,1,figsize=(12,8))
axes2.bar(string, frames, color='blue')
axes2.set_ylabel('Seconds, s', size = 18)
axes2.set_xticklabels(rotation=0, labels = string)
axes2.grid(True)
axes2.tick_params(axis='both', which='major', labelsize=14)
fig2.show()
fig2.savefig("Compare_Models/Direct_Multi_Step_Probability_Results/Figures/Training_Times.pdf", bbox_inches='tight')

# Might have to use subplots for the large timescales. https://stackoverflow.com/questions/5656798/python-matplotlib-is-there-a-way-to-make-a-discontinuous-axis