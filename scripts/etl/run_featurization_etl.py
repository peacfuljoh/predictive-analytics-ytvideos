"""Script for experimenting with ETL pipeline"""

from src.crawler.crawler.config import DB_MONGO_CONFIG
from src.etl_pipelines.featurization_etl import etl_features_main
from src.etl_pipelines.featurization_etl_utils import (ETLRequestFeatures, DB_FEATURES_NOSQL_DATABASE,
                                                       DB_FEATURES_NOSQL_COLLECTIONS)
from src.etl_pipelines.etl_request import validate_etl_config


ETL_CONFIG_VALID_KEYS_FEATURES = dict(
    extract=['filters', 'limit'],
    transform=[],
    load=[]
)
ETL_CONFIG_EXCLUDE_KEYS_FEATURES = dict(
    extract=['filters', 'limit'],
    transform=[],
    load=[]
)


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

req = ETLRequestFeatures(etl_config,
                         etl_config_name,
                         ETL_CONFIG_VALID_KEYS_FEATURES,
                         ETL_CONFIG_EXCLUDE_KEYS_FEATURES)
validate_etl_config(req,
                    DB_MONGO_CONFIG,
                    DB_FEATURES_NOSQL_DATABASE,
                    DB_FEATURES_NOSQL_COLLECTIONS['etl_config_features'])

etl_features_main(req)
