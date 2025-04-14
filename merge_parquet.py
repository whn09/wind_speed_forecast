import os
import time
import pandas as pd
from tqdm import tqdm
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# 新疆边界
lat_min, lat_max = 34, 49
lon_min, lon_max = 73, 96

def load_parquet(filename, horizon=1):
    df = pd.read_parquet(filename)
    df = df[(df['latitude']>=lat_min) & (df['latitude']<=lat_max) & (df['longitude']>=lon_min) & (df['longitude']<=lon_max)]
    df['item_id'] = df['latitude'].astype(str)+'_'+df['longitude'].astype(str)
    df.drop(['latitude', 'longitude'], axis=1, inplace=True)
    
    # 根据不同的horizon值筛选特定的时间点
    if horizon == 24:
        # 只保留0点的数据
        df = df[df['time'].dt.hour == 0]
    elif horizon == 6:
        # 只保留0, 6, 12, 18点的数据
        df = df[df['time'].dt.hour.isin([0, 6, 12, 18])]
    elif horizon == 3:
        # 只保留0, 3, 6, 9, 12, 15, 18, 21点的数据
        df = df[df['time'].dt.hour.isin([0, 3, 6, 9, 12, 15, 18, 21])]
    # 对于horizon=1或其他值，保留所有时间点的数据
    
    # data = TimeSeriesDataFrame.from_data_frame(
    #     df,
    #     id_column="item_id",
    #     timestamp_column="time"
    # )
    return df

if __name__ == '__main__':
    base_dir = '/opt/dlami/nvme/surface/'
    horizon = 24  # 24, 6, 3, 1
    
    # 获取所有parquet文件的列表
    filenames = [f for f in os.listdir(base_dir) if f.endswith('parquet')]
    
    # 创建一个空的TimeSeriesDataFrame或普通DataFrame来存储合并后的数据
    all_data = None
    
    print(f"处理{len(filenames)}个parquet文件，horizon={horizon}...")
    
    # 遍历所有parquet文件并合并数据
    for filename in tqdm(filenames[:2]):
        file_path = os.path.join(base_dir, filename)
        start = time.time()
        data = load_parquet(file_path, horizon=horizon)
        end = time.time()
        print('load_parquet time:', end-start)
        
        # 如果all_data为空，初始化它
        if all_data is None:
            all_data = data
        else:
            all_data = pd.concat([all_data, data])
    
    # 保存合并后的数据到一个新的parquet文件
    output_filename = f"merged_data_horizon_{horizon}.parquet"
    output_path = os.path.join(base_dir, output_filename)
    
    all_data.to_parquet(output_path)
    
    print(f"合并完成，保存到: {output_path}")
    print(f"合并后数据形状: {all_data.shape}")