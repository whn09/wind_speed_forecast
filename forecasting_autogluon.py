import os
import time
import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.metrics import mean_squared_error

horizon = 6  # 24,6,3,1
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
    time_limit=3600,  # recommended: 3600
)

# predictor = TimeSeriesPredictor.load(f"autogluon-wind-speed-horizon-{horizon}")

# predictions = predictor.predict(train_data)
# print('predictions:', predictions)

# 按时间间隔评估RMSE
start_date = test_df['time'].min()
end_date = test_df['time'].max()
# end_date = test_df['time'].min()+pd.Timedelta(hours=48)  # TODO only for unit test
current_time = start_date
current_df = train_df.copy()

daily_rmse = {}
time_step = pd.Timedelta(hours=horizon)  # 根据horizon动态设置时间步长

# 记录所有日期，用于后续绘图
all_dates = []

while current_time <= end_date:
    next_time = current_time + time_step
    
    # 获取当前时间段的测试数据
    interval_test_df = test_df[(test_df['time'] >= current_time) & (test_df['time'] < next_time)]
    
    if len(interval_test_df) > 0:
        start = time.time()
        # 准备当前的训练数据
        current_test_data = TimeSeriesDataFrame.from_data_frame(
            current_df,
            id_column="item_id",
            timestamp_column="time"
        )
        end = time.time()
        
        # 获取预测
        interval_predictions = predictor.predict(current_test_data)
        end2 = time.time()
        
        # 确保预测结果和测试数据可以对齐
        # 注意：实际应用中可能需要更复杂的对齐逻辑
        pred_values = interval_predictions['mean'].values
        test_values = interval_test_df['wind_speed'].values
        
        # 如果预测结果和测试数据长度不匹配，可能需要截断
        min_length = min(len(pred_values), len(test_values))
        pred_values = pred_values[:min_length]
        test_values = test_values[:min_length]
        
        if min_length > 0:
            rmse = np.sqrt(mean_squared_error(test_values, pred_values))
            date_key = current_time.date()  # 使用日期作为键
            
            # 如果同一日期有多个时间点，取平均值
            if date_key in daily_rmse:
                daily_rmse[date_key].append(rmse)
            else:
                daily_rmse[date_key] = [rmse]
                all_dates.append(date_key)
            
            print(f"Time: {current_time}, RMSE: {rmse}")
        
        # 更新当前数据集，添加这个时间段的测试数据
        current_df = pd.concat([current_df, interval_test_df])
    
    current_time = next_time

# 计算每日平均RMSE
avg_daily_rmse = {date: np.mean(rmses) for date, rmses in daily_rmse.items()}

# 绘制每日RMSE图表
plt.figure(figsize=(15, 6))
dates = all_dates
rmse_values = [avg_daily_rmse[date] for date in dates]
plt.plot(dates, rmse_values, marker='o')
plt.title(f'Daily RMSE for Test Data (Horizon = {horizon}h)')
plt.xlabel('Date')
plt.ylabel('RMSE')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'daily_rmse_horizon_{horizon}.png')

# 保存每日RMSE到CSV文件
daily_rmse_df = pd.DataFrame({
    'date': dates,
    'rmse': rmse_values
})
daily_rmse_df.to_csv(f'daily_rmse_horizon_{horizon}.csv', index=False)