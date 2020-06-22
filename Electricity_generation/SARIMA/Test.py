import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'G'


import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import math

# Import the timeseries data and convert the strings to floats
# This csv data has missing values for the first 151 days of the year. (Get rid of them)
df18 = pd.read_csv(
    "/Users/benoitputzeys/PycharmProjects/MSc_Thesis/Data_Entsoe/Total_Load_Country/Total Load - Day Ahead _ Actual_201801010000-201901010000.csv")
df18['Actual Total Load [MW] - United Kingdom (UK)'] = df18['Actual Total Load [MW] - United Kingdom (UK)'].astype(        float)
df18 = df18.truncate(before=151 * 48)

df19 = pd.read_csv(
    "/Users/benoitputzeys/PycharmProjects/MSc_Thesis/Data_Entsoe/Total_Load_Country/Total Load - Day Ahead _ Actual_201901010000-202001010000.csv")
df19['Actual Total Load [MW] - United Kingdom (UK)'] = df19['Actual Total Load [MW] - United Kingdom (UK)'].astype(        float)

# This csv data has missing values for the last 152 days of the year as they lie in the future. (Get rid of them)
df20 = pd.read_csv(
    "/Users/benoitputzeys/PycharmProjects/MSc_Thesis/Data_Entsoe/Total_Load_Country/Total Load - Day Ahead _ Actual_202001010000-202101010000.csv")
df20.loc[df20['Actual Total Load [MW] - United Kingdom (UK)'] == '-', 'Actual Total Load [MW] - United Kingdom (UK)'] = 0
df20['Actual Total Load [MW] - United Kingdom (UK)'] = df20['Actual Total Load [MW] - United Kingdom (UK)'].astype(float)
df20 = df20.truncate(after=152*48-1)

frames = ([df18, df19, df20])
df_raw = pd.concat(frames)

# Determine the Features.
df_features = pd.DataFrame()

# Create the date.
df_features["Date"]=df_raw["Time (CET)"]

for i in range(len(df_features)):
    df_features.iloc[i, 0] = pd.to_datetime([df_features.iloc[i, 0][0:16]], format='%d.%m.%Y %H:%M')[0]
# Determine the load.
df_features["Total_Load_Past"] = df_raw["Actual Total Load [MW] - United Kingdom (UK)"]

# Create your input variable
df = df_features
df = df.drop(df.index[0:2])

df = df[:500]

y = df.set_index(['Date'])
y.head(5)
y.plot(figsize=(19, 4))
plt.show()

y['Total_Load_Past'] = y['Total_Load_Past'].fillna(0)

from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y, model='additive',period = 48)
fig = decomposition.plot()
plt.show()
#
# from pylab import rcParams
# rcParams['figure.figsize'] = 18, 8
# decomposition = sm.tsa.seasonal_decompose(df, model='additive')
# fig = decomposition.plot()
# plt.show()

