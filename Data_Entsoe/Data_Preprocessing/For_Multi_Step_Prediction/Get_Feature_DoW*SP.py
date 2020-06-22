import pandas as pd
import numpy as np

X = pd.read_csv("/Users/benoitputzeys/PycharmProjects/MSc_Thesis/Data_Entsoe/Data_Preprocessing/X.csv")

DoW_2 = 0
SP_2 = 0
i=len(X)
j = 6
k = 5
while i <= (len(X)+48*70-1):
    if k==49:
        k=1
        j=j+1
        if j==7:
            j=0
    DoW_2 = np.append(DoW_2, j)
    SP_2 = np.append(SP_2, k)
    i=i+1
    k=k+1

SP_2 = np.delete(SP_2,0)
DoW_2 = np.delete(DoW_2,0)

DoW_SP_2 = DoW_2*SP_2
DoW_SP_2 = DoW_SP_2.reshape(len(DoW_SP_2),1)

np.savetxt("/Users/benoitputzeys/PycharmProjects/MSc_Thesis/Data_Entsoe/Data_Preprocessing/For_Multi_Step_Prediction/DoW_SP_2.csv", DoW_SP_2, delimiter=",")
