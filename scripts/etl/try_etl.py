"""Script for experimenting with ETL pipeline"""

import pandas as pd

from src.preprocessor.prefeaturization_etl import etl_main
from src.preprocessor.prefeaturization_etl_utils import ETLRequest
from src.visualization.dashboard import Dashboard


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

req = ETLRequest(etl_config)

data = etl_main(req)

dfs = []
while not (df_ := next(data)).empty:
    dfs.append(df_)
df = pd.concat(dfs, axis=0, ignore_index=True)

if 1:
    dashboard = Dashboard(df)
    dashboard.run()
