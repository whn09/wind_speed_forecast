# Wind Speed Forecast

python extract_wind_speed_from_era5.py

python merge_parquet.py

pip install uv
uv pip install -q autogluon.timeseries --system
uv pip uninstall -q torchaudio torchvision torchtext --system # fix incompatible package versions on Colab

pip install --upgrade flash-attn --no-build-isolation

python forecasting_autogluon.py

s5cmd cp s3://datalab/nsf-ncar-era5/surface/wind_speed_*.parquet /opt/dlami/nvme/nsf-ncar-era5/surface/
