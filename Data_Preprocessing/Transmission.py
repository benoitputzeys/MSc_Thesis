from entsoe import EntsoePandasClient
import pandas as pd
import datetime as dt

client = EntsoePandasClient(api_key="b7a8a6e4-3d85-427a-8790-30ab56538691")
start = pd.Timestamp('20160101', tz="Europe/London")
end = pd.Timestamp('20200615', tz="Europe/London")

start_BE = pd.Timestamp('20180101', tz="Europe/London")
end_BE = pd.Timestamp('20200615', tz="Europe/London")

load_GB_raw = client.query_load("GB", start = start ,end = end)
load_GB_raw.to_csv("Data_Preprocessing/Load_GB_Raw_Data")

transmission_BE_to_GB_raw = client.query_crossborder_flows(country_code_to="GB", country_code_from="BE" ,start = start_BE ,end = end_BE)
transmission_FR_to_GB_raw = client.query_crossborder_flows(country_code_to="GB", country_code_from="FR" ,start = start ,end = end)
transmission_IE_to_GB_raw = client.query_crossborder_flows(country_code_to="GB", country_code_from="IE" ,start = start ,end = end)
transmission_NL_to_GB_raw = client.query_crossborder_flows(country_code_to="GB", country_code_from="NL" ,start = start ,end = end)

transmission_GB_to_BE_raw = client.query_crossborder_flows(country_code_to="BE", country_code_from="GB" ,start = start_BE ,end = end_BE)
transmission_GB_to_FR_raw = client.query_crossborder_flows(country_code_to="FR", country_code_from="GB" ,start = start ,end = end)
transmission_GB_to_IE_raw = client.query_crossborder_flows(country_code_to="IE", country_code_from="GB" ,start = start ,end = end)
transmission_GB_to_NL_raw = client.query_crossborder_flows(country_code_to="NL", country_code_from="GB" ,start = start ,end = end)

# Fill NaN values with 0.
transmission_BE_to_GB_raw = transmission_BE_to_GB_raw.fillna(0)
transmission_FR_to_GB_raw = transmission_FR_to_GB_raw.fillna(0)
transmission_IE_to_GB_raw = transmission_IE_to_GB_raw.fillna(0)
transmission_NL_to_GB_raw = transmission_NL_to_GB_raw.fillna(0)

# Fill NaN values with 0.
transmission_GB_to_BE_raw = transmission_GB_to_BE_raw.fillna(0)
transmission_GB_to_FR_raw = transmission_GB_to_FR_raw.fillna(0)
transmission_GB_to_IE_raw = transmission_GB_to_IE_raw.fillna(0)
transmission_GB_to_NL_raw = transmission_GB_to_NL_raw.fillna(0)

# Calculate the net electricity load going into GB.
transmission_BE_GB_raw = transmission_BE_to_GB_raw - transmission_GB_to_BE_raw
transmission_FR_GB_raw = transmission_FR_to_GB_raw - transmission_GB_to_FR_raw
transmission_IE_GB_raw = transmission_IE_to_GB_raw - transmission_GB_to_IE_raw
transmission_NL_GB_raw = transmission_NL_to_GB_raw - transmission_GB_to_NL_raw

# Import all the values into a dataframe and compute the net total.
dataframe = pd.DataFrame(data = {"BE_GB": transmission_BE_GB_raw,
             "FR_GB": transmission_FR_GB_raw,
             "IE_GB": transmission_IE_GB_raw,
             "NL_GB": transmission_NL_GB_raw,})
dataframe = dataframe.fillna(0)
dataframe["Total"] = dataframe["BE_GB"] + dataframe["FR_GB"] + dataframe["IE_GB"] + dataframe["NL_GB"]

# Increase the frequency of the indexing.
dataframe = dataframe.resample('30min').pad()/2

# Import the processed load.
load = pd.read_csv("Data_Preprocessing/Load_GB_Processed_Data")
load = load.rename(columns={"Unnamed: 0": "Timestamp", "0": "Load"}, errors="raise")

load["Timestamp"] = [dt.datetime.strptime(load.iloc[i,0][0:16], '%Y-%m-%d %H:%M') for i in range(len(load))]
load = load.set_index(["Timestamp"])

load = load.tz_localize(None)
dataframe = dataframe.tz_localize(None)

# Get rid of duplicative data.
load = load.loc[~load.index.duplicated(keep='first')]
dataframe = dataframe.loc[~dataframe.index.duplicated(keep='first')]

# Save the data in a new dataframe.
load_and_transmission = pd.DataFrame(data = {"Load": load["Load"], "Transmission": dataframe["Total"]})
load_and_transmission = load_and_transmission.fillna(method='ffill')

load_and_transmission.to_csv("Data_Preprocessing/Load_and_Transmission_Data.csv")

# truevals = pd.read_csv("Data_Preprocessing/Load_GB_Processed_Data")
# truevals = truevals.rename(columns={"Unnamed: 0": "Timestamp", "0": "Load"}, errors="raise")
#
# truevals["Timestamp"] = [dt.datetime.strptime(truevals.iloc[i,0][0:16], '%Y-%m-%d %H:%M') for i in range(len(truevals))]
# truevals = truevals.set_index(["Timestamp"])
#
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(2,1,figsize=(10,10))
# ax[0].plot(load_and_transmission.iloc[:,1], color = "blue", linewidth = 0.5)
# ax[1].plot(load_and_transmission.iloc[:,0]-truevals.iloc[:,0], color = "red", linewidth = 1)
# fig.show()

