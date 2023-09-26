"""Script for experimenting with ETL pipeline"""

from src.preprocessor.featurization_etl import etl_main
from src.preprocessor.featurization_etl_utils import ETLRequest
from src.visualization.dashboard import Dashboard


etl_config = {
    'extract': {
        'filters': {
            # 'timestamp_accessed': [['2023-09-10 00:00:00.000', '2024-01-01 00:00:00.000']],
            'username': ['CNN', "TheYoungTurks", "FoxNews", "WashingtonPost", "msnbc", "NBCNews"]
        },
        # 'limit': 1000
    }
}

req = ETLRequest(etl_config)

data = etl_main(req)


if 0:
    dashboard = Dashboard(data['stats'])
    dashboard.run()
