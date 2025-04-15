import os
import time
import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

horizon = 24  # 24,6,3,1
train_test_split_time = pd.Timestamp('2024-01-01 00:00:00')
base_dir = '/opt/dlami/nvme/surface/'
lead_time = 10

# 计算预测长度，以天数转换为预测步数
prediction_length = int(lead_time * 24 / horizon)
print(f"Prediction length: {prediction_length} steps (equivalent to {lead_time} days)")

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
    prediction_length=prediction_length,  # 1,
    path=f"autogluon-wind-speed-horizon-{horizon}",
    target="wind_speed",
    eval_metric="RMSE",
)

predictor.fit(
    train_data,
    presets="high_quality",
    time_limit=3600,  # recommended: 3600
)

# predictor.fit(
#     train_data,
#     presets="bolt_base",
# )

# predictor.fit(
#     train_data,
#     hyperparameters={
#         "Chronos": [
#             {"model_path": "bolt_base", "ag_args": {"name_suffix": "ZeroShot"}},
#             {"model_path": "bolt_small", "fine_tune": True, "ag_args": {"name_suffix": "FineTuned"}},
#         ]
#     },
#     time_limit=3600,  # time limit in seconds
#     enable_ensemble=False)

# predictor = TimeSeriesPredictor.load(f"autogluon-wind-speed-horizon-{horizon}")

# predictions = predictor.predict(train_data)
# print('predictions:', predictions)

# 按天评估RMSE
start_date = test_df['time'].min()
end_date = test_df['time'].max() - pd.Timedelta(days=lead_time)
# end_date = start_date+pd.Timedelta(days=3)  # sample
print(f"Evaluation period: {start_date} to {end_date}")

# 存储每一天预测的每一个未来天的RMSE
daily_future_rmse = {}  # 格式: {预测日期: {未来第几天: rmse值}}
all_dates = []

# 计算总天数用于进度条
current_date = start_date.floor('D')  # 确保从当天的00:00开始
total_days = (end_date - current_date).days + 1

# 创建进度条
with tqdm(total=total_days, desc="Evaluating predictions") as pbar:
    # 按天进行预测评估
    while current_date <= end_date:
        # 更新进度条的描述，显示当前日期
        pbar.set_description(f"Evaluating {current_date.date()}")
        
        # 准备截至当前日期的所有数据作为训练集
        start = time.time()
        current_train = df[df['time'] < current_date]
        current_train_data = TimeSeriesDataFrame.from_data_frame(
            current_train,
            id_column="item_id",
            timestamp_column="time"
        )
        end = time.time()
        print('create TimeSeriesDataFrame time:', end-start)
        
        # 获取预测结果
        predictions = predictor.predict(current_train_data)
        end2 = time.time()
        print('predict time:', end2-end)
        
        # 计算每个未来天的RMSE
        future_rmse = {}
        for future_day in range(1, lead_time + 1):
            # 计算未来第几天的开始和结束时间
            future_start = current_date + pd.Timedelta(days=future_day-1)
            future_end = current_date + pd.Timedelta(days=future_day)
            
            # 获取这几天的实际数据
            actual_data = test_df[(test_df['time'] >= future_start) & (test_df['time'] < future_end)]
            
            if not actual_data.empty:
                # 找出对应的预测数据
                # 计算预测结果中对应的索引范围
                start_idx = (future_day - 1) * (24 // horizon)
                end_idx = future_day * (24 // horizon)
                
                # 确保索引不超出预测结果范围
                end_idx = min(end_idx, len(predictions) // len(predictions.item_ids))
                
                # 只处理有足够预测步数的情况
                if start_idx < end_idx:
                    # 获取所有item_id的预测值合并为一个列表
                    all_preds = []
                    all_actuals = []
                    
                    for item_id in predictions.item_ids:
                        # 获取该item_id的预测
                        item_preds = predictions.loc[item_id, 'mean'].values[start_idx:end_idx]
                        
                        # 获取该item_id的实际值
                        item_actuals = actual_data[actual_data['item_id'] == item_id]['wind_speed'].values
                        
                        # 如果两者长度不同，取最小的长度
                        min_len = min(len(item_preds), len(item_actuals))
                        if min_len > 0:
                            all_preds.extend(item_preds[:min_len])
                            all_actuals.extend(item_actuals[:min_len])
                    
                    if all_preds and all_actuals:
                        # 计算RMSE
                        rmse = np.sqrt(mean_squared_error(all_actuals, all_preds))
                        future_rmse[future_day] = rmse
        
        # 保存当天的结果
        if future_rmse:
            date_key = current_date.date()
            daily_future_rmse[date_key] = future_rmse
            if date_key not in all_dates:
                all_dates.append(date_key)
        
        # 前进到下一天
        current_date += pd.Timedelta(days=1)
        
        # 更新进度条
        pbar.update(1)

# 计算每个预测天的平均RMSE
avg_future_day_rmse = {}
for future_day in range(1, lead_time + 1):
    day_rmses = [day_data.get(future_day, np.nan) for day_data in daily_future_rmse.values()]
    day_rmses = [rmse for rmse in day_rmses if not np.isnan(rmse)]
    if day_rmses:
        avg_future_day_rmse[future_day] = np.mean(day_rmses)

# 输出每个未来天的平均RMSE
print("\nAverage RMSE by forecast day:")
for day, rmse in avg_future_day_rmse.items():
    print(f"  Day {day}: {rmse:.4f}")

# 绘制平均未来天RMSE图表
plt.figure(figsize=(12, 6))
future_days = list(avg_future_day_rmse.keys())
rmse_values = [avg_future_day_rmse[day] for day in future_days]
plt.plot(future_days, rmse_values, marker='o')
plt.title(f'Average RMSE by Forecast Day (Horizon = {horizon}h)')
plt.xlabel('Days into the Future')
plt.ylabel('RMSE')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'avg_future_day_rmse_horizon_{horizon}.png')

# 绘制每个预测日期的RMSE热图
if all_dates and daily_future_rmse:
    plt.figure(figsize=(15, 8))
    
    # 创建数据矩阵
    rmse_matrix = np.full((len(all_dates), lead_time), np.nan)
    for i, date in enumerate(all_dates):
        for day in range(1, lead_time + 1):
            if date in daily_future_rmse and day in daily_future_rmse[date]:
                rmse_matrix[i, day-1] = daily_future_rmse[date][day]
    
    # 绘制热图
    plt.imshow(rmse_matrix, aspect='auto', cmap='viridis')
    plt.colorbar(label='RMSE')
    plt.title(f'RMSE by Prediction Date and Future Day (Horizon = {horizon}h)')
    plt.xlabel('Days into the Future')
    plt.ylabel('Prediction Date')
    plt.xticks(range(0, lead_time), range(1, lead_time + 1))
    plt.yticks(range(0, len(all_dates), max(1, len(all_dates)//10)), 
               [all_dates[i].strftime('%Y-%m-%d') for i in range(0, len(all_dates), max(1, len(all_dates)//10))])
    plt.tight_layout()
    plt.savefig(f'rmse_heatmap_horizon_{horizon}.png')

# 保存详细结果到CSV
results = []
for date in all_dates:
    for future_day in range(1, lead_time + 1):
        if date in daily_future_rmse and future_day in daily_future_rmse[date]:
            results.append({
                'prediction_date': date,
                'forecast_day': future_day,
                'rmse': daily_future_rmse[date][future_day]
            })

results_df = pd.DataFrame(results)
results_df.to_csv(f'future_rmse_details_horizon_{horizon}.csv', index=False)

# 保存每个未来天的平均RMSE
avg_results = [{'forecast_day': day, 'avg_rmse': rmse} 
               for day, rmse in avg_future_day_rmse.items()]
avg_results_df = pd.DataFrame(avg_results)
avg_results_df.to_csv(f'avg_future_day_rmse_horizon_{horizon}.csv', index=False)

print("Evaluation complete. Results saved to CSV files and plots.")