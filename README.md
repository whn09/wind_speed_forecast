# Wind Speed Forecast

python extract_wind_speed_from_era5.py

python merge_parquet.py

pip install uv
uv pip install autogluon.timeseries --system
uv pip uninstall torchaudio torchvision torchtext --system # fix incompatible package versions on Colab

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

pip install --upgrade flash-attn --no-build-isolation
pip install tf-keras

python forecasting_autogluon.py

s5cmd cp s3://datalab/nsf-ncar-era5/surface/wind_speed_*.parquet /opt/dlami/nvme/surface/
s5cmd cp s3://datalab/nsf-ncar-era5/surface/merged_data_horizon_*.parquet /opt/dlami/nvme/surface/
