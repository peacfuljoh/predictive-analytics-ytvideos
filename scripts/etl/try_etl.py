"""Script for experimenting with ETL pipeline"""

from src.preprocessor.preprocessor import ETLRequest, etl_main
from src.crawler.crawler.utils.misc_utils import print_df_full



etl_config = {
    'extract': {
        'filters': {
            'timestamp_accessed': ['2023-09-20 00:00:00.000', '2023-09-20 2:00:00.000'],
            'username': 'CNN'
        },
        'limit': 1000
    }
}

req = ETLRequest(etl_config)

data = etl_main(req)

print_df_full(data)