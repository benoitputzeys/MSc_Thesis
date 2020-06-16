
def return_features_and_labels():

    import pandas as pd
    import numpy as np
    from sklearn.impute import SimpleImputer

    # Import the timeseries data
    df = pd.read_csv("/Users/benoitputzeys/PycharmProjects/NN-Predicitons/Data/SPI_202005.csv")
    # df4 = pd.read_csv("/Users/benoitputzeys/PycharmProjects/NN-Predicitons/Data/SPI_202004.csv")
    # df3 = pd.read_csv("/Users/benoitputzeys/PycharmProjects/NN-Predicitons/Data/SPI_202003.csv")
    # df2 = pd.read_csv("/Users/benoitputzeys/PycharmProjects/NN-Predicitons/Data/SPI_202002.csv")
    # df1 = pd.read_csv("/Users/benoitputzeys/PycharmProjects/NN-Predicitons/Data/SPI_202001.csv")
    #
    # frames = ([df1, df2, df3, df4, df5])
    # df = pd.concat(frames)
    #
    # Determine the label
    df_label = df["Total_Generation"]

    # Determine the Features
    df_features = pd.DataFrame()
    df_features["Total_Generation"] = df["Total_Generation"].shift(+2)
    df_features["Settlement_Period"] = df["Settlement_Period"]
    df_features["System_Buy_Price"] = df["System_Buy_Price"].shift(+2)
    df_features["Total_Demand"] = df["Total_Demand"].shift(+2)
    df_features["Total_Period_Applicable_Balancing_Services_Volume"] = df["Total_Period_Applicable_Balancing_Services_Volume"].shift(+2)
    df_features["Total_NIV_Tagged_Volume"] = df["Total_NIV_Tagged_Volume"].shift(+2)
    df_features["System_Total_Accepted_Bid_Volume"] = df["System_Total_Accepted_Bid_Volume"].shift(+2)
    df_features["Total_System_Accepted_Offer_Volume"] = df["Total_System_Accepted_Offer_Volume"].shift(+2)
    df_features["Total_System_Tagged_Accepted_Bid_Volume"] = df["Total_System_Tagged_Accepted_Bid_Volume"].shift(+2)
    df_features["Total_System_Tagged_Accepted_Offer_Volume"] = df["Total_System_Tagged_Accepted_Offer_Volume"].shift(+2)

    # Create your input variable
    X = df_features.values
    y = df_label.values

    y = np.reshape(y, (len(y), 1))

    # After having shifted the data, the nan values have to be replaces in order to have good predictions.
    replace_nan = SimpleImputer(missing_values=np.nan, strategy='mean')
    replace_nan.fit(X)
    X = replace_nan.transform(X)

    return X, y