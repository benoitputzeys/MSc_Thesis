import pandas as pd
import matplotlib.pyplot as plt

AF_ML = pd.read_csv("Electricity_Generation_Prediction\ANN\Result_Features_Length\All_Features_Max_Length.csv")
AF_But_Trans_ML = pd.read_csv("Electricity_Generation_Prediction\ANN\Result_Features_Length\All_But_Transmission_Features_Max_Length.csv")
AF_HL = pd.read_csv("Electricity_Generation_Prediction\ANN\Result_Features_Length\All_Features_Half_Length.csv")
AF_Quarter_L = pd.read_csv("Electricity_Generation_Prediction\ANN\Result_Features_Length\All_Features_Quarter_Length.csv")
AF_35_L = pd.read_csv("Electricity_Generation_Prediction\ANN\Result_Features_Length\All_Features_35_Length.csv")
AF_34_L = pd.read_csv("Electricity_Generation_Prediction\ANN\Result_Features_Length\All_Features_Three_Quarter_Length.csv")
AF_25_L = pd.read_csv("Electricity_Generation_Prediction\ANN\Result_Features_Length\All_Features_25_Length.csv")

frames = ([AF_But_Trans_ML, AF_ML,AF_34_L,AF_25_L, AF_HL, AF_35_L, AF_Quarter_L])
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
