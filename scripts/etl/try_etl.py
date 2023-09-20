"""Script for experimenting with ETL pipeline"""

from src.preprocessor.preprocessor import ETLRequest, etl_main
from src.visualization.dashboard import Dashboard


etl_config = {
    'extract': {
        'filters': {
            # 'timestamp_accessed': ['2023-09-10 00:00:00.000', '2024-01-01 00:00:00.000'],
            # 'username': 'CNN'
        },
        # 'limit': 1000
    }
}

req = ETLRequest(etl_config)

data = etl_main(req)


if 1:
    dashboard = Dashboard(data['stats'])
    dashboard.run()
