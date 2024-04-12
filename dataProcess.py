import pandas as pd
import torch
import math
from torch.nn.utils.rnn import pad_sequence
from datetime import datetime, timedelta
from torch.utils.data import Dataset, DataLoader


def dataToTensorHourly(path, separateByDay=True, missingThreshold=0.1, columnToDelete=['wind_dir', 'unixtime'], start=None, end=(datetime.now().date())):
    """
    Takes the relative path to an hourly weather csv file and returns a tensor

    :param str path: The relative path of the hourly weather csv to be parsed
    :param bool separateByDay: Whether or not to output multiple tensors for each day
    :param float missingThreshold: Removes columns with a missing data ratio greather than this value
    :param list columnToDelete: Names of columns to remove
    :param datetime.date start: Only count data from this date
    :param datetime.date end: Only count data before this date
    """
    df = pd.read_csv(path)
    df = df.rename(columns={'date_time_local':'hour'})
    df['day'] = None
    df['day_since_beginning'] = None
    if start is None:
        start = datetime.strptime(df.iloc[-1]['hour'], '%Y-%m-%d %H:%M:%S EST')
        start = start.date()
    for index, row in df.iterrows():
        if pd.isna(df.at[index, 'pressure_station']):
            df = df.drop(index)
        else:
            try:
                date = datetime.strptime(row['hour'], '%Y-%m-%d %H:%M:%S EDT')
            except:
                date = datetime.strptime(row['hour'], '%Y-%m-%d %H:%M:%S EST')
            if date.date() > end or date.date() < start:
                df = df.drop(index)
            else:
                df.at[index, 'day_since_beginning'] = int((date.date() - start).days)
                df.at[index,'hour'] = int(date.hour)
                df.at[index,'day'] = int((date - datetime.strptime(str(date.year), "%Y")).days)
    if columnToDelete is not None:
        df = df.drop(labels=columnToDelete, axis=1)
    for i in list(df.columns.values):
        if df[i].isna().sum() / df.shape[0] > missingThreshold:
            df = df.drop(labels=i, axis=1)
    if missingThreshold > 0:
        df.interpolate()
    # print(df)
    if separateByDay:
        tensors = []
        group = df.groupby('day_since_beginning')
        for _, c in group:
            # print(_)
            c = c.drop(labels='day_since_beginning', axis=1)
            tensors.insert(0, torch.tensor(c.to_numpy().astype(float)))
        return tensors
    df = df.drop(labels='day_since_beginning', axis=1)
    return [torch.tensor(df.to_numpy().astype(float))]

def dailyTargets(path, target='avg_temperature', start=None, end=datetime.now().date()):
    df = pd.read_csv(path)
    if start is None:
        start = datetime.strptime(df.iloc[-1]['date'], '%Y-%m-%d')
        start = start.date()
    for index, row in df.iterrows():
        date = datetime.strptime(row['date'], "%Y-%m-%d").date()
        if date > end or date < start:
            df = df.drop(index)
    return(torch.tensor(df[target].to_numpy().astype(float)))

class dataSet(Dataset):
    def __init__(self, hourly_path, daily_path, start, end):
        data_end = (datetime.combine(end, datetime.min.time()) - timedelta(1)).date()
        target_start = (datetime.combine(start, datetime.min.time()) + timedelta(1)).date()
        self.data = dataToTensorHourly(hourly_path, start=start, end=data_end)
        self.targets = dailyTargets(daily_path, start=target_start, end=end)
        
    def __len__(self):
        return self.targets.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

if __name__ == "__main__":
    # Set the start and end times based on your csv files
    start1 = datetime(2021, 4, 13).date()
    end2 = datetime(2024, 4, 10).date()

    # Generate a dataset
    data = dataSet('.\\Raw data\\three_year\\weatherstats_toronto_hourly.csv', ".\\Raw data\\three_year\\weatherstats_toronto_daily.csv", start1, end2)

    # Set up a 60/20/20 train val test split
    train = data[:math.floor(len(data) * .6)]
    validation = data[math.floor(len(data) * .6) + 1 : math.floor(len(data) * .8)]
    test = data[math.floor(len(data) * .8 + 1) :]

    # Example dataloader call
    train_dataloader = DataLoader(train, batch_size=10)