# Wind Speed Forecast

python extract_wind_speed_from_era5.py

python merge_parquet.py

pip install uv
uv pip install -q autogluon.timeseries --system
uv pip uninstall -q torchaudio torchvision torchtext --system # fix incompatible package versions on Colab

python forecasting_autogluon.py

s5cmd cp s3://datalab/nsf-ncar-era5/surface/wind_speed_*.parquet /opt/dlami/nvme/nsf-ncar-era5/surface/


        Failed to import transformers.integrations.integration_utils because of the following error (look up to see its traceback):
Failed to import transformers.modeling_utils because of the following error (look up to see its traceback):
/opt/pytorch/lib/python3.12/site-packages/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so: undefined symbol: _ZN3c105ErrorC2ENS_14SourceLocationENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
Training timeseries model ChronosFineTuned[bolt_small]. Training for up to 14.3s of the 85.7s of remaining time.
        Skipping covariate_regressor since the dataset contains no covariates or static features.
        Warning: Exception caused ChronosFineTuned[bolt_small] to fail during training... Skipping this model.
        Failed to import transformers.integrations.integration_utils because of the following error (look up to see its traceback):
Failed to import transformers.modeling_utils because of the following error (look up to see its traceback):
/opt/pytorch/lib/python3.12/site-packages/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so: undefined symbol: _ZN3c105ErrorC2ENS_14SourceLocationENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE

