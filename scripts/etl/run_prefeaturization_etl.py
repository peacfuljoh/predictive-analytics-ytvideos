"""Script for experimenting with ETL pipeline"""

import pandas as pd

from src.crawler.crawler.config import DB_MONGO_CONFIG
from src.etl_pipelines.prefeaturization_etl import etl_prefeatures_main
from src.etl_pipelines.prefeaturization_etl_utils import (ETLRequestPrefeatures, DB_FEATURES_NOSQL_DATABASE,
                                                          DB_FEATURES_NOSQL_COLLECTIONS)
from src.etl_pipelines.etl_request import validate_etl_config
from src.visualization.dashboard import Dashboard



ETL_CONFIG_VALID_KEYS_PREFEATURES = dict(
    extract=['filters', 'limit'],
    transform=['include_additional_keys'],
    load=[],
    preconfig=[]
)
ETL_CONFIG_EXCLUDE_KEYS_PREFEATURES = dict(
    extract=['filters', 'limit'],
    transform=[],
    load=[],
    preconfig=[]
)




etl_config_name = 'test3'
# etl_config_name = 'dashboard'

if etl_config_name == 'test3':
    etl_config = {
        'extract': {
            'filters': {
                # 'timestamp_accessed': [['2023-09-10 00:00:00.000', '2024-01-01 00:00:00.000']],
                'username': ['CNN', "TheYoungTurks", "FoxNews", "WashingtonPost", "msnbc", "NBCNews"]
            },
            # 'limit': 1000
        }
    }
if etl_config_name == 'dashboard':
    etl_config = {
        'extract': {
            'filters': {
                # 'timestamp_accessed': [['2023-09-10 00:00:00.000', '2024-01-01 00:00:00.000']],
                'username': ['CNN', "TheYoungTurks", "FoxNews", "WashingtonPost", "msnbc", "NBCNews"]
            },
            # 'limit': 1000
        },
        'transform': {
            'include_additional_keys': ['title', 'upload_date', 'timestamp_first_seen']
        }
    }

req = ETLRequestPrefeatures(etl_config,
                            etl_config_name,
                            ETL_CONFIG_VALID_KEYS_PREFEATURES,
                            ETL_CONFIG_EXCLUDE_KEYS_PREFEATURES)
validate_etl_config(req,
                    DB_MONGO_CONFIG,
                    DB_FEATURES_NOSQL_DATABASE,
                    DB_FEATURES_NOSQL_COLLECTIONS['etl_config_prefeatures'])

data = etl_prefeatures_main(req, return_for_dashboard=etl_config_name == 'dashboard')

if etl_config_name == 'dashboard':
    dfs = []
    while not (df_ := next(data)).empty:
        dfs.append(df_)
    df = pd.concat(dfs, axis=0, ignore_index=True)

    dashboard = Dashboard(df)
    dashboard.run()
