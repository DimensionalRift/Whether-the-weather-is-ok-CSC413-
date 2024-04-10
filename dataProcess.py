import pandas as pd
import torch
import numpy as np
from datetime import datetime

def dataToTensorHourly(path, missingThreshold=0.1, columnToDelete=['wind_dir', 'unixtime']):
    """
    Takes the relative path to an hourly weather csv file and returns a tensor

    :param str path: The relative path of the hourly weather csv to be parsed
    :param float missingThreshold: Removes columns with a missing data ratio greather than this value
    :param columnToDelete list: Names of columns to remove
    """
    df = pd.read_csv(path)
    df = df.rename(columns={'date_time_local':'time_local'})
    df['date_local'] = None
    for index, row in df.iterrows():
        if pd.isna(df.at[index, 'pressure_station']):
            df = df.drop(index)
        else:
            try:
                date = datetime.strptime(row['time_local'], '%Y-%m-%d %H:%M:%S EDT')
            except:
                date = datetime.strptime(row['time_local'], '%Y-%m-%d %H:%M:%S EST')
            df.at[index,'time_local'] = int(date.hour)
            df.at[index,'date_local'] = int((date - datetime.strptime(str(date.year), "%Y")).days)
    if columnToDelete is not None:
        df = df.drop(labels=columnToDelete, axis=1)
    for i in list(df.columns.values):
        if df[i].isna().sum() / df.shape[0] > missingThreshold:
            df = df.drop(labels=i, axis=1)
    if missingThreshold > 0:
        df.interpolate()
    print(df)
    return torch.tensor(df.to_numpy().astype(float))



if __name__ == "__main__":
    data = dataToTensorHourly(".\Raw data\weatherstats_toronto_hourly.csv", missingThreshold=.2)
    print(data)
    print(data.shape)
    print(data[0])