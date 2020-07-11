from entsoe import EntsoePandasClient
import pandas as pd
import matplotlib.pyplot as plt

client = EntsoePandasClient(api_key="b7a8a6e4-3d85-427a-8790-30ab56538691")
start = pd.Timestamp('20160101', tz="Europe/London")
end = pd.Timestamp('20200615', tz="Europe/London")
cc = "GB"

load_UK_raw = client.query_load(cc, start = start ,end = end)
load_UK_raw.to_csv("Data_Entsoe/Data_Preprocessing/For_Multi_Step_Prediction/Load_UK_Raw_Data")

# Unavailabilities needed?
unavailability_UK = client.query_unavailability_of_generation_units(cc, start = start ,end = end)

# Plot raw data.
fig1, axs1=plt.subplots(1,1,figsize=(12,6))
axs1.plot(load_UK_raw)
axs1.set_ylabel("Load in GB [MW]")
axs1.set_xlabel("Date")
fig1.suptitle("Unprocessed data", fontsize=18, x = 0.52, y = 0.975)
fig1.show()


load_UK_processed = load_UK_raw.copy()

# Filter some erroneous data out.
for i in range(1,len(load_UK_processed)-1):
    if (load_UK_processed[i] > load_UK_processed[i - 1] * 1.5) & (load_UK_processed[i] > load_UK_processed[i + 1] * 1.5):
        load_UK_processed[i] = load_UK_processed[i - 1]
    if (load_UK_processed[i] < load_UK_processed[i - 1] / 1.2) & (load_UK_processed[i] < load_UK_processed[i + 1] / 1.2):
            load_UK_processed[i] = load_UK_processed[i - 1]


# Create plot of processed data.
fig2, axs2=plt.subplots(1,1,figsize=(12,6))
axs2.plot(load_UK_processed)
axs2.set_ylabel("Load in GB [MW]")
axs2.set_xlabel("Date")
fig2.show()

# Create plot of 2018/12/15 unusual load.
fig3, axs3=plt.subplots(1,1,figsize=(12,6))
axs3.plot(load_UK_raw[51600:52000])
axs3.set_ylabel("Load in GB [MW]")
axs3.set_xlabel("Date")
fig3.show()

# Create plot of 2018/12/15 processed load.
fig4, axs4=plt.subplots(1,1,figsize=(12,6))
load_UK_processed[51772:51784]=load_UK_processed[51772-48:51784-48]
axs4.plot(load_UK_processed[51600:52000])
axs4.set_ylabel("Load in GB [MW]")
axs4.set_xlabel("Date")
fig4.show()

# Create plot of 2019/06/23 processed load.
fig5, axs5=plt.subplots(1,1,figsize=(12,6))
axs5.plot(load_UK_processed[60800:61000])
axs5.set_ylabel("Load in GB [MW]")
axs5.set_xlabel("Date")
fig5.show()

# Create plot of 2019/06/23 processed load.
fig6, axs6=plt.subplots(1,1,figsize=(12,6))
load_UK_processed[60894:60896]=load_UK_raw[60894-48*7:60896-48*7]
axs6.plot(load_UK_processed[60800:61000])
axs6.set_ylabel("Load in GB [MW]")
axs6.set_xlabel("Date")
fig6.show()

# Create plot of 2019/06/13 unusual load.
fig7, axs7=plt.subplots(1,1,figsize=(12,6))
axs7.plot(load_UK_raw[60200:60600])
axs7.set_ylabel("Load in GB [MW]")
axs7.set_xlabel("Date")
fig7.show()

# Create plot of 2019/06/13 processed load.
fig8, axs8=plt.subplots(1,1,figsize=(12,6))
load_UK_processed[60397:60423]=load_UK_raw[60397-48:60423-48]
axs8.plot(load_UK_processed[60200:60600])
axs8.set_ylabel("Load in GB [MW]")
axs8.set_xlabel("Date")
fig8.show()

# Create plot of 2019/07/25 & 2019/07/26 unprocessed load.
fig9, axs9=plt.subplots(1,1,figsize=(12,6))
axs9.plot(load_UK_raw[62200:62650])
axs9.set_ylabel("Load in GB [MW]")
axs9.set_xlabel("Date")
fig9.show()

# Create plot of 2019/07/25 processed load.
fig10, axs10=plt.subplots(1,1,figsize=(12,6))
load_UK_processed[62431:62439]= load_UK_raw[62431-48*7:62439-48*7]
axs10.plot(load_UK_processed[62200:62650])
axs10.set_ylabel("Load in GB [MW]")
axs10.set_xlabel("Date")
fig10.show()

# Create plot of 2019/07/26 processed load.
fig11, axs11=plt.subplots(1,1,figsize=(12,6))
load_UK_processed[62382:62391]= load_UK_raw[62382-48*7:62391-48*7]
axs11.plot(load_UK_processed[62200:62650])
axs11.set_ylabel("Load in GB [MW]")
axs11.set_xlabel("Date")
fig11.show()

# Create plot of 2019/05/21 unprocessed load.
fig12, axs12=plt.subplots(1,1,figsize=(12,6))
axs12.plot(load_UK_raw[59000:59750])
axs12.set_ylabel("Load in GB [MW]")
axs12.set_xlabel("Date")
fig12.show()

# Create plot of 2019/05/21 processed load.
fig13, axs13=plt.subplots(1,1,figsize=(12,6))
load_UK_processed[59273:59280] = load_UK_processed[59273-46:59280-46]
axs13.plot(load_UK_processed[59000:59750])
axs13.set_ylabel("Load in GB [MW]")
axs13.set_xlabel("Date")
fig13.show()

# Create plot of 2019/03/19 unprocessed load.
fig14, axs14=plt.subplots(1,1,figsize=(12,6))
#load_UK_processed[59275:59282] = load_UK_processed[59275-46:59282-46]
axs14.plot(load_UK_raw[56100:56600])
axs14.set_ylabel("Load in GB [MW]")
axs14.set_xlabel("Date")
fig14.show()

# Create plot of 2019/03/19 processed load.
fig15, axs15=plt.subplots(1,1,figsize=(12,6))
load_UK_processed[56272:56274] = load_UK_processed[56272-48*7:56274-48*7]
axs15.plot(load_UK_processed[56100:56600])
axs15.set_ylabel("Load in GB [MW]")
axs15.set_xlabel("Date")
fig15.show()

# Create plot of 2019/03/28 unprocessed load.
fig16, axs16=plt.subplots(1,1,figsize=(12,6))
axs16.plot(load_UK_raw[56540:56941])
axs16.set_ylabel("Load in GB [MW]")
axs16.set_xlabel("Date")
fig16.show()

# Create plot of 2019/03/28 processed load.
fig17, axs17=plt.subplots(1,1,figsize=(12,6))
load_UK_processed[56738:56739] = load_UK_processed[56738-48:56739-48]
axs17.plot(load_UK_processed[56540:56941])
axs17.set_ylabel("Load in GB [MW]")
axs17.set_xlabel("Date")
fig17.show()

# Create plot of 2019/03/20 unprocessed load.
fig18, axs18=plt.subplots(1,1,figsize=(12,6))
axs18.plot(load_UK_raw[56300:56400])
axs18.set_ylabel("Load in GB [MW]")
axs18.set_xlabel("Date")
fig18.show()

# Create plot of 2019/03/20 processed load.
fig19, axs19=plt.subplots(1,1,figsize=(12,6))
load_UK_processed[56370:56372] = load_UK_raw[56370-48:56372-48]
axs19.plot(load_UK_processed[56300:56400])
axs19.set_ylabel("Load in GB [MW]")
axs19.set_xlabel("Date")
fig19.show()

# Create plot of all the processed load.
fig20, axs20=plt.subplots(1,1,figsize=(12,6))
axs20.plot(load_UK_processed)
axs20.set_ylabel("Load in GB [MW]")
axs20.set_xlabel("Date")
fig20.suptitle("Processed data", fontsize=18, x = 0.52, y = 0.975)
fig20.show()

load_UK_processed.to_csv("Data_Entsoe/Data_Preprocessing/For_Multi_Step_Prediction/Load_UK_Processed_Data")



