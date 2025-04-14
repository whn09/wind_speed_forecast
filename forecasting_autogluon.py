import os
import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import matplotlib.pyplot as plt
from datetime import timedelta

horizon = 24  # 24,6,3,1
train_test_split_time = pd.Timestamp('2024-01-01 00:00:00')
base_dir = '/opt/dlami/nvme/surface/'

df = pd.read_parquet(os.path.join(base_dir, f'merged_data_horizon_{horizon}.parquet'))
print('df:', df)

train_df = df[df['time']<train_test_split_time]
test_df = df[df['time']>=train_test_split_time]

train_data = TimeSeriesDataFrame.from_data_frame(
    train_df,
    id_column="item_id",
    timestamp_column="time"
)

print('train_data:', train_data)

# predictor = TimeSeriesPredictor(
#     freq=f'{horizon}h',
#     prediction_length=1,
#     path=f"autogluon-wind-speed-horizon-{horizon}",
#     target="wind_speed",
#     eval_metric="RMSE",
# )

# predictor.fit(
#     train_data,
#     presets="high_quality",
#     time_limit=120,
# )

predictor = TimeSeriesPredictor.load(f"autogluon-wind-speed-horizon-{horizon}")

# predictions = predictor.predict(train_data)
# print('predictions:', predictions)

# 按天评估RMSE
start_date = test_df['time'].min().date()
end_date = test_df['time'].max().date()
current_date = start_date
current_df = train_df

daily_rmse = {}

while current_date <= end_date:
    next_date = current_date + timedelta(days=1)
    
    # 获取当天的测试数据
    daily_test_df = test_df[(test_df['time'].dt.date == current_date)]
    
    if len(daily_test_df) > 0:
        current_df = pd.concat([current_df, daily_test_df])
        
        daily_test_data = TimeSeriesDataFrame.from_data_frame(
            current_df,
            id_column="item_id",
            timestamp_column="time"
        )
        # print('daily_test_data:', daily_test_data)
        
        # # 获取预测
        # daily_predictions = predictor.predict(daily_test_data)
        # print('daily_predictions:', daily_predictions)
        
        # 计算当天的RMSE
        score = predictor.evaluate(daily_test_data)  # TODO
        daily_rmse[current_date] = score['RMSE']
        
        print(f"Date: {current_date}, RMSE: {daily_rmse[current_date]}")
    
    current_date = next_date

# 绘制每日RMSE图表
plt.figure(figsize=(15, 6))
dates = list(daily_rmse.keys())
rmse_values = list(daily_rmse.values())
plt.plot(dates, rmse_values, marker='o')
plt.title('Daily RMSE for Test Data')
plt.xlabel('Date')
plt.ylabel('RMSE')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('daily_rmse.png')
# plt.show()

# 保存每日RMSE到CSV文件
daily_rmse_df = pd.DataFrame({
    'date': dates,
    'rmse': rmse_values
})
daily_rmse_df.to_csv(f'daily_rmse_horizon_{horizon}.csv', index=False)
