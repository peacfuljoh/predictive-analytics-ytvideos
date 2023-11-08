"""Script for experimenting with ETL pipeline"""

import pandas as pd

from src.etl.prefeaturization_etl import etl_prefeatures_main
from src.etl.etl_request_utils import get_validated_etl_request
from src.visualization.dashboard import Dashboard
from src.crawler.crawler.constants import (COL_UPLOAD_DATE, COL_TIMESTAMP_FIRST_SEEN, COL_USERNAME,
                                           COL_TIMESTAMP_ACCESSED, COL_TITLE)
from src.crawler.crawler.config import DB_INFO, DB_MYSQL_CONFIG, DB_MONGO_CONFIG



# setup config
etl_config_name = 'test3'
# etl_config_name = 'dashboard'

db_ = {'db_info': DB_INFO, 'db_mysql_config': DB_MYSQL_CONFIG, 'db_mongo_config': DB_MONGO_CONFIG}

if etl_config_name == 'test3':
    etl_config = {
        'extract': {
            'filters': {
                # COL_TIMESTAMP_ACCESSED: [['2023-09-10 00:00:00.000', '2024-01-01 00:00:00.000']],
                COL_USERNAME: ['CNN', "TheYoungTurks", "FoxNews", "WashingtonPost", "msnbc", "NBCNews"]
            },
            # 'limit': 1000
        },
        'db': db_
    }
if etl_config_name == 'dashboard':
    etl_config = {
        'extract': {
            'filters': {
                # COL_TIMESTAMP_ACCESSED: [['2023-09-10 00:00:00.000', '2024-01-01 00:00:00.000']],
                COL_USERNAME: ['CNN', "TheYoungTurks", "FoxNews", "WashingtonPost", "msnbc", "NBCNews"]
            },
            # 'limit': 1000
        },
        'transform': {
            'include_additional_keys': [COL_TITLE, COL_UPLOAD_DATE, COL_TIMESTAMP_FIRST_SEEN]
        },
        'db': db_
    }

# make ETL request object
req = get_validated_etl_request('prefeatures', etl_config, etl_config_name)

# run pipeline
data = etl_prefeatures_main(req, return_for_dashboard=etl_config_name == 'dashboard')

# run dashboard
if etl_config_name == 'dashboard':
    dfs = [df for df in data] # TODO: make this on-demand
    df = pd.concat(dfs, axis=0, ignore_index=True)

    dashboard = Dashboard(df)
    dashboard.run()
