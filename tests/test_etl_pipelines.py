"""Script for experimenting with ETL pipeline"""

from src.crawler.crawler.constants import COL_USERNAME
from constants_tests import db_info, ETL_CONFIG_NAME_PREFEATURES_TEST
from utils_for_tests import setup_for_prefeatures_tests, verify_prefeatures_tests


def test_prefeaturization_etl_pipeline():
    from src.etl.prefeaturization_etl import etl_prefeatures_main
    from src.etl.prefeaturization_etl_utils import get_etl_req_prefeats

    # setup request
    etl_config = {
        'extract': {
            'filters': {
                COL_USERNAME: ['username1', 'username2']
            },
        },
        'db': db_info
    }
    req = get_etl_req_prefeats(ETL_CONFIG_NAME_PREFEATURES_TEST, etl_config)

    # run test
    setup_for_prefeatures_tests(req) # setup for test
    etl_prefeatures_main(req) # run pipeline
    verify_prefeatures_tests(req) # verify results

