import pickle
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import torch

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
import matplotlib.pyplot as plt

from datetime import datetime

from lightning.pytorch.tuner import Tuner

start = datetime(2014, 1, 3).date()
end = datetime(2024, 4, 10).date()
hourly_path ='.\\Raw data\\ten_year\\weatherstats_toronto_hourly.csv'
daily_path =  '.\\Raw data\\ten_year\\weatherstats_toronto_daily.csv'

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

def generateTimeSeriesDataset(path, missingThreshold=0.1, columnToDelete=['wind_dir', 'unixtime'], start=None, end=(datetime.now().date())):
  
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
        cols = df.columns
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
        df.infer_objects(copy=False)
        df = df.interpolate(axis=1)
    
    # Drop NAs
    df = df.dropna()
    # Set constant column for the group input
    # As we only have this one timeseries
    df['group']=0

    df['total_hours'] = (df['day_since_beginning'] * 24) + df['hour']
    df['total_hours'] = df['total_hours'].astype(int)
    df = df.sort_values('total_hours')
    
    # Hours
    max_prediction_length = 72
    test_reservation = 72
    training_cutoff = df["total_hours"].max() - max_prediction_length - test_reservation
    val_cutoff = df["total_hours"].max() - test_reservation
    
    max_encoder_length= 72

    training = TimeSeriesDataSet(
    df[df['total_hours'] < training_cutoff],
    group_ids=["group"],
    target="temperature",
    time_idx="total_hours",
    min_encoder_length=max_encoder_length//2,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    time_varying_unknown_reals=['pressure_station', 'pressure_sea', 'wind_dir_10s',
       'wind_speed', 'relative_humidity', 'dew_point', 'temperature',
       'visibility', 'cloud_cover_8', 'max_air_temp_pst1hr',
       'min_air_temp_pst1hr'],
    time_varying_known_reals=["total_hours"],
    allow_missing_timesteps=True,
    add_relative_time_idx=True,
    add_target_scales=False,
    add_encoder_length=True,
    )
    
    validation = TimeSeriesDataSet.from_dataset(training, df[df['total_hours'] < val_cutoff], predict=True, stop_randomization=True)
    
    test = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)
    
    
    return training, validation, test

def generateDataloader(dataset, train=True, batch_size=128):
    
    dataloader = dataset.to_dataloader(train=train, batch_size=batch_size, num_workers=0)
  
    return dataloader

def train(training, validation, batch_size=128):
    
    # create dataloaders for model
    # set batch size to between 32 to 128
    train_dataloader = generateDataloader(training, True, batch_size)
    val_dataloader =  generateDataloader(validation, False, batch_size)

    # setup trainer and transformer
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()  
    logger = TensorBoardLogger("lightning_logs")  

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="auto",
        enable_model_summary=True,
        gradient_clip_val=0.1,
        limit_train_batches=50,  
        callbacks=[lr_logger, early_stop_callback],
        logger=logger,
    )

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=8,
        loss=QuantileLoss(),
        log_interval=10, 
        optimizer="Ranger",
        reduce_on_plateau_patience=4,
    )
    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")


    # fit network
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


    # Hyperparam optimizing

    study = optimize_hyperparameters(
        train_dataloader,
        val_dataloader,
        model_path="optuna_test",
        n_trials=200, #200
        max_epochs=50, #50
        gradient_clip_val_range=(0.01, 1.0),
        hidden_size_range=(8, 128),
        hidden_continuous_size_range=(8, 128),
        attention_head_size_range=(1, 4),
        learning_rate_range=(0.001, 0.1),
        dropout_range=(0.1, 0.3),
        trainer_kwargs=dict(limit_train_batches=30),
        reduce_on_plateau_patience=4,
        use_learning_rate_finder=False,  # use Optuna to find ideal learning rate or use in-built learning rate finder
    )

    with open("test_study.pkl", "wb") as fout:
        pickle.dump(study, fout)


    print(study.best_trial.params)
    #{'gradient_clip_val': 0.18288910107905082, 'hidden_size': 41, 'dropout': 0.2657743937910419, 'hidden_continuous_size': 14, 'attention_head_size': 3, 'learning_rate': 0.026593399882835468}

    best_model_path = trainer.checkpoint_callback.best_model_path
    print(best_model_path)
    return best_model_path

def genValPrediction(model_path, validation, batch_size=128):
    
    val_dataloader = generateDataloader(validation, False, batch_size)
    
    best_tft = TemporalFusionTransformer.load_from_checkpoint(model_path)


    predictions = best_tft.predict(val_dataloader, return_y=True, trainer_kwargs=dict(accelerator="cpu"))
    print(MAE()(predictions.output, predictions.y))

    raw_predictions = best_tft.predict(val_dataloader, mode="raw", return_x=True)


    predict_plot = best_tft.plot_prediction(raw_predictions.x, raw_predictions.output, idx=0, add_loss_to_title=True)
    predict_plot.savefig('predict_plot_val.png')
    
    interpretation = best_tft.interpret_output(raw_predictions.output, reduction="sum")
    plots = best_tft.plot_interpretation(interpretation)
    for key, value in plots.items():
        value.savefig(f"{key}.png")
    
def genPrediction(model_path, dataset, batch_size=128):
    
    dataloader = generateDataloader(dataset, False, batch_size)
    
    best_tft = TemporalFusionTransformer.load_from_checkpoint(model_path)


    predictions = best_tft.predict(dataloader, return_y=True, trainer_kwargs=dict(accelerator="cpu"))
    print(MAE()(predictions.output, predictions.y))

    raw_predictions = best_tft.predict(dataloader, mode="raw", return_x=True)


    predict_plot = best_tft.plot_prediction(raw_predictions.x, raw_predictions.output, idx=0, add_loss_to_title=True)
    predict_plot.savefig('predict_plot.png')
    
def main():
    
    pl.seed_everything(42)
    training, validation, test = generateTimeSeriesDataset(hourly_path, start=start, end=end)
    best_model_path = train(training, validation, 128)
    #best_model_path = 'lightning_logs\\lightning_logs\\version_1\\checkpoints\\epoch=35-step=1800.ckpt'
    
    genValPrediction(best_model_path, validation, 128)
    
    genPrediction(best_model_path, test, 128)
    

if __name__ == "__main__":
    main()
    
    
    