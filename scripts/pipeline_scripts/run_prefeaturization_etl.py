"""Script for experimenting with ETL pipeline"""

import pandas as pd

from src.etl.prefeaturization_etl import etl_prefeatures_main
from src.etl.prefeaturization_etl_utils import get_etl_req_prefeats
from src.visualization.dashboard import Dashboard
from src.crawler.crawler.constants import (COL_UPLOAD_DATE, COL_TIMESTAMP_FIRST_SEEN, COL_USERNAME,
                                           COL_TIMESTAMP_ACCESSED, COL_TITLE)


# setup config
etl_config_name = 'test3'
# etl_config_name = 'dashboard'

if etl_config_name == 'test3':
    etl_config = {
        'extract': {
            'filters': {
                # COL_TIMESTAMP_ACCESSED: [['2023-09-10 00:00:00.000', '2024-01-01 00:00:00.000']],
                COL_USERNAME: ['CNN', "TheYoungTurks", "FoxNews", "WashingtonPost", "msnbc", "NBCNews"]
            },
            # 'limit': 1000
        }
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
        }
    }

# make ETL request object
req = get_etl_req_prefeats(etl_config_name, etl_config)

# run pipeline
data = etl_prefeatures_main(req, return_for_dashboard=etl_config_name == 'dashboard')

# run dashboard
if etl_config_name == 'dashboard':
    dfs = [df for df in data] # TODO: make this on-demand
    df = pd.concat(dfs, axis=0, ignore_index=True)

    dashboard = Dashboard(df)
    dashboard.run()
