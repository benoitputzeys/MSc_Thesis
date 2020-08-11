from entsoe import EntsoePandasClient
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

client = EntsoePandasClient(api_key="b7a8a6e4-3d85-427a-8790-30ab56538691")
start = pd.Timestamp('20160101', tz="Europe/London")
end = pd.Timestamp('20200615', tz="Europe/London")
cc = "GB"

load_GB_raw = client.query_load(cc, start = start ,end = end)
load_GB_raw.to_csv("Data_Preprocessing/Load_GB_Raw_Data")

# # Unavailabilities needed?
# unavailability_GB = client.query_unavailability_of_generation_units(cc, start = start ,end = end)

# Plot raw data.
fig1, axs1=plt.subplots(1,1,figsize=(12,6))
axs1.plot(load_GB_raw/1000, color = "blue", linewidth = 0.5)
axs1.set_ylabel("Load in GB (Raw Data), GW",size = 16)
axs1.set_xlabel("Date", size = 16)
axs1.axes.tick_params(labelsize = 14)
axs1.grid(True)
#fig1.suptitle("Unprocessed data", fontsize=18, x = 0.52, y = 0.975)
fig1.show()
fig1.savefig("Data_Preprocessing/Figures/Raw_Data.pdf", bbox_inches='tight')

load_GB_processed = load_GB_raw.copy()

not_shifted_raw = np.array(load_GB_raw)
shifted_raw = load_GB_processed.shift(+1)
not_shifted = not_shifted_raw[1:]
shifted_raw = shifted_raw[1:]

# Plot the difference in electricity load from one SP to the next.
fig21, axs21=plt.subplots(1,1,figsize=(12,6))
axs21.plot((shifted_raw-not_shifted)/1000, color = "blue", linewidth = 0.5)
axs21.set_ylabel("Difference of Load, GW",size = 16)
axs21.set_xlabel("Date", size = 16)
axs21.axes.tick_params(labelsize = 14)
axs21.grid(True)
fig21.show()

# Filter some erroneous data out.
for i in range(1,len(load_GB_processed)-1):
    if (np.abs(load_GB_processed[i] - load_GB_processed[i - 1]) > 10000) & (np.abs(load_GB_processed[i] - load_GB_processed[i + 1])>10000):
        load_GB_processed[i] = load_GB_processed[i - 1]

# Create plot of processed data.
fig2, axs2=plt.subplots(1,1,figsize=(12,6))
axs2.plot(load_GB_processed/1000, color = "blue", linewidth = 0.5)
axs2.set_ylabel("Load in GB (Processed Data), GW",size = 16)
axs2.set_xlabel("Date", size = 16)
axs2.axes.tick_params(labelsize = 14)
axs2.grid(True)
fig2.show()

# Create plot of 2018/12/15 unusual load.
fig3, axs3=plt.subplots(2,1,figsize=(12,6))
axs3[0].plot(load_GB_raw[51600:52000]/1000, color = "blue")
axs3[0].set_ylabel("Raw Data, GW",size = 16)
axs3[0].axes.tick_params(labelsize = 12)
axs3[0].grid(True)

load_GB_processed[51772:51784]=load_GB_processed[51772-48:51784-48]
axs3[1].plot(load_GB_processed[51600:52000]/1000, color = "blue")
axs3[1].set_ylabel("Processed Data, GW",size = 16)
axs3[1].set_xlabel("Date", size = 16)
axs3[1].axes.tick_params(labelsize = 12)
axs3[1].grid(True)
fig3.show()

# Create plot of 2019/06/23 processed load.
fig4, axs4=plt.subplots(2,1,figsize=(12,6))
axs4[0].plot(load_GB_raw[60800:61000]/1000, color = "blue")
axs4[0].set_ylabel("Raw Data, GW",size = 16)
axs4[0].axes.tick_params(labelsize = 12)
axs4[0].grid(True)

load_GB_processed[60894:60896]=load_GB_raw[60894-48*7:60896-48*7]
axs4[1].plot(load_GB_processed[60800:61000]/1000, color = "blue")
axs4[1].set_ylabel("Processed Data, GW",size = 16)
axs4[1].set_xlabel("Date", size = 16)
axs4[1].axes.tick_params(labelsize = 12)
axs4[1].grid(True)
fig4.show()

# Create plot of 2019/06/13 unusual load.
fig5, axs5=plt.subplots(2,1,figsize=(12,6))
axs5[0].plot(load_GB_raw[60200:60600]/1000, color = "blue")
axs5[0].set_ylabel("Raw Data, GW",size = 16)
axs5[0].axes.tick_params(labelsize = 12)
axs5[0].grid(True)

load_GB_processed[60397:60423]=load_GB_raw[60397-48:60423-48]
axs5[1].plot(load_GB_processed[60200:60600]/1000, color = "blue")
axs5[1].set_ylabel("Processed Data, GW",size = 16)
axs5[1].set_xlabel("Date",size = 16)
axs5[1].axes.tick_params(labelsize = 12)
axs5[1].grid(True)
fig5.show()

# Create plot of 2019/07/25 & 2019/07/26 unprocessed load.
fig6, axs6=plt.subplots(3,1,figsize=(12,6))
axs6[0].plot(load_GB_raw[62200:62650]/1000, color = "blue")
axs6[0].set_ylabel("Raw Data, GW", size = 12)
axs6[0].axes.tick_params(labelsize = 12)
axs6[0].grid(True)

load_GB_processed[62431:62439]= load_GB_raw[62431-48*7:62439-48*7]
axs6[1].plot(load_GB_processed[62200:62650]/1000, color = "blue")
axs6[1].set_ylabel("Processed Data, GW", size = 12)
axs6[1].axes.tick_params(labelsize = 12)
axs6[1].grid(True)

load_GB_processed[62382:62391]= load_GB_raw[62382-48*7:62391-48*7]
axs6[2].plot(load_GB_processed[62200:62650]/1000, color = "blue")
axs6[2].set_ylabel("Processed Data, GW", size = 12)
axs6[2].set_xlabel("Date", size = 16)
axs6[2].axes.tick_params(labelsize = 12)
axs6[2].grid(True)
fig6.show()

# Create plot of 2019/05/21 unprocessed load.
fig7, axs7=plt.subplots(2,1,figsize=(12,6))
axs7[0].plot(load_GB_raw[59000:59750]/1000, color = "blue")
axs7[0].set_ylabel("Raw Data, GW", size = 16)
axs7[0].axes.tick_params(labelsize = 12)
axs7[0].grid(True)

load_GB_processed[59273:59280] = load_GB_processed[59273-46:59280-46]
axs7[1].plot(load_GB_processed[59000:59750]/1000, color = "blue")
axs7[1].set_ylabel("Processed Data, GW", size = 16)
axs7[1].set_xlabel("Date", size = 16)
axs7[1].axes.tick_params(labelsize = 12)
axs7[1].grid(True)
fig7.show()
fig7.savefig("Data_Preprocessing/Figures/Example_Preprocessing.pdf", bbox_inches='tight')

# Create plot of 2019/03/19 unprocessed load.
fig8, axs8=plt.subplots(3,1,figsize=(12,6))
axs8[0].plot(load_GB_raw[56100:56600]/1000, color = "blue")
axs8[0].set_ylabel("Raw Data, GW", size = 12)
axs8[0].axes.tick_params(labelsize = 11)
axs8[0].grid(True)

load_GB_processed[56272:56274] = load_GB_processed[56272-48*7:56274-48*7]
axs8[1].plot(load_GB_processed[56100:56600]/1000, color = "blue")
axs8[1].set_ylabel("Processed Data, GW", size = 12)
axs8[1].axes.tick_params(labelsize = 11)
axs8[1].grid(True)

load_GB_processed[56370:56372] = load_GB_raw[56370-48:56372-48]
axs8[2].plot(load_GB_processed[56100:56600]/1000, color = "blue")
axs8[2].set_ylabel("Processed Data, GW", size = 12)
axs8[2].set_xlabel("Date", size = 18)
axs8[1].axes.tick_params(labelsize = 11)
axs8[2].grid(True)
fig8.show()

# Create plot of 2019/03/28 unprocessed load.
fig9, axs9=plt.subplots(2,1,figsize=(12,6))
axs9[0].plot(load_GB_raw[56540:56941]/1000, color = "blue")
axs9[0].set_ylabel("Raw Data, GW", size = 16)
axs9[1].axes.tick_params(labelsize = 11)
axs9[0].grid(True)

load_GB_processed[56738:56739] = load_GB_processed[56738-48:56739-48]
axs9[1].plot(load_GB_processed[56540:56941]/1000, color = "blue")
axs9[1].set_ylabel("Processed Data, GW", size = 16)
axs9[1].set_xlabel("Date", size = 16)
axs9[1].axes.tick_params(labelsize = 11)
axs9[1].grid(True)
fig9.show()

# Create plot of all the processed load.
fig10, axs10=plt.subplots(1,1,figsize=(12,6))
axs10.plot(load_GB_processed/1000, color = "blue", linewidth = 0.5)
axs10.set_ylabel("Load in GB (Processed Data), GW", size = 18)
axs10.set_xlabel("Date", size = 18)
axs10.axes.tick_params(labelsize = 12)
#fig20.suptitle("Processed data", fontsize=18, x = 0.52, y = 0.975)
axs10.grid(True)
fig10.show()
fig10.savefig("Data_Preprocessing/Figures/Processed_Data.pdf", bbox_inches='tight')

load_GB_processed.to_csv("Data_Preprocessing/Load_GB_Processed_Data")

not_shifted_processed = np.array(load_GB_processed)
shifted_processed = load_GB_processed.shift(+1)
not_shifted_processed = not_shifted_processed[1:]
shifted_processed = shifted_processed[1:]

# Make 2 subplots to show the differences in the SP again.
fig21, axs21=plt.subplots(2,1,figsize=(12,6))
axs21[0].plot((shifted_raw-not_shifted)/1000, color = "blue", linewidth = 0.5)
axs21[0].set_ylabel("Difference\n(Raw Data), GW", size = 16)
axs21[1].plot((shifted_processed-not_shifted_processed)/1000, color = "blue", linewidth = 0.5)
axs21[1].set_ylabel("Difference\n(Processed Data), GW", size = 16)
axs21[1].set_xlabel("Date", size = 16)
axs21[0].axes.tick_params(labelsize = 12)
axs21[1].axes.tick_params(labelsize = 12)
axs21[0].grid(True)
axs21[1].grid(True)
fig21.show()
fig21.savefig("Data_Preprocessing/Figures/Difference_Load_Before_and_After.pdf", bbox_inches='tight')


