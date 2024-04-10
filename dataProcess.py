import pandas as pd
import torch

def dataToTensorHourly(path, missingThreshold=0.1, columnToDelete=['wind_dir', 'date_time_local']):
    """
    Takes the relative path to an hourly weather csv file and returns a tensor

    :param str path: The relative path of the hourly weather csv to be parsed
    :param float missingThreshold: Removes columns with a missing data ratio greather than this value
    :param columnToDelete list: Names of columns to remove
    """
    df = pd.read_csv(path)
    for index, _ in df.iterrows():
        if pd.isna(df.at[index, 'pressure_station']):
            df = df.drop(index)
    if columnToDelete is not None:
        df = df.drop(labels=columnToDelete, axis=1)
    for i in list(df.columns.values):
        if df[i].isna().sum() / df.shape[0] > missingThreshold:
            df = df.drop(labels=i, axis=1)
    # df = df.reset_index()
    if missingThreshold > 0:
        df.interpolate()
    print(df)
    return torch.tensor(df.values)

if __name__ == "__main__":
    print(dataToTensorHourly(".\Raw data\weatherstats_toronto_hourly.csv", missingThreshold=.2).shape)