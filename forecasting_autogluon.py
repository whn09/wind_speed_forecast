import os
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import matplotlib.pyplot as plt

horizon = 24  # 24,6,3,1
train_test_split_time = pd.Timestamp('2024-01-01 00:00:00')
base_dir = '/opt/dlami/nvme/surface/'

data = pd.read_parquet(os.path.join(base_dir, f'merged_data_horizon_{horizon}.parquet'))
print('data:', data)

train_data = data[data['time']<train_test_split_time]
test_data = data[data['time']>=train_test_split_time]

train_data = TimeSeriesDataFrame.from_data_frame(
    train_data,
    id_column="item_id",
    timestamp_column="time"
)

test_data = TimeSeriesDataFrame.from_data_frame(
    test_data,
    id_column="item_id",
    timestamp_column="time"
)

print('train_data:', train_data)
print('test_data:', test_data)

predictor = TimeSeriesPredictor(
    freq=f'{horizon}h',
    prediction_length=1,
    path=f"autogluon-wind-speed-horizon-{horizon}",
    target="wind_speed",
    eval_metric="RMSE",
)

predictor.fit(
    train_data,
    presets="high_quality",
    time_limit=3600,
)

# predictor = TimeSeriesPredictor.load("autogluon-wind-speed-hourly")

predictions = predictor.predict(train_data)
print('predictions:', predictions)

# Plot 4 randomly chosen time series and the respective forecasts
# predictor.plot(test_data[(test_data.index.get_level_values('timestamp')>=predictions.index.get_level_values('timestamp').min()) & (test_data.index.get_level_values('timestamp')<=predictions.index.get_level_values('timestamp').max())], predictions, quantile_levels=[0.1, 0.9], max_history_length=200, max_num_item_ids=4)

print(predictor.leaderboard(test_data))
