"""Script for experimenting with ETL pipeline"""

import pandas as pd

from src.etl_pipelines.featurization_etl import etl_main
from src.etl_pipelines.featurization_etl_utils import ETLRequestFeatures, verify_valid_features_etl_config


etl_config_name = 'test'

if etl_config_name == 'test':
    etl_config = {
        'extract': {
            'filters': {
                # 'timestamp_accessed': [['2023-09-10 00:00:00.000', '2024-01-01 00:00:00.000']],
                'username': ['CNN', "TheYoungTurks", "FoxNews", "WashingtonPost", "msnbc", "NBCNews"]
            },
            # 'limit': 1000
        }
    }

req = ETLRequestFeatures(etl_config, etl_config_name)
verify_valid_features_etl_config(req)

data = etl_main(req)
