"""Script for experimenting with ETL pipeline"""

import pandas as pd

from src.etl.prefeaturization_etl import etl_prefeatures_main
from src.etl.prefeaturization_etl_utils import get_etl_req_prefeats
from src.visualization.dashboard import Dashboard


# setup config
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

# make ETL request object
req = get_etl_req_prefeats(etl_config_name, etl_config)

# run pipeline
data = etl_prefeatures_main(req, return_for_dashboard=etl_config_name == 'dashboard')

# run dashboard
if etl_config_name == 'dashboard':
    dfs = []
    while not (df_ := next(data)).empty:
        dfs.append(df_)
    df = pd.concat(dfs, axis=0, ignore_index=True)

    dashboard = Dashboard(df)
    dashboard.run()
