"""Script for experimenting with ETL pipeline"""

import os

os.environ['RUN_TESTS'] = 'yes' # always set to  'yes' for tests!

from src.crawler.crawler.constants import COL_USERNAME



def test_prefeaturization_etl_pipeline():
    from src.etl.prefeaturization_etl import etl_prefeatures_main
    # from src.etl.prefeaturization_etl_utils import get_etl_req_prefeats
    from src.etl.etl_request_utils import get_validated_etl_request
    from src.crawler.crawler.config import DB_INFO, ETL_CONFIG_NAME_PREFEATURES_TEST
    from utils_for_tests import setup_for_prefeatures_tests, verify_prefeatures_tests

    # # setup request
    # etl_config = {
    #     'extract': {
    #         'filters': {
    #             COL_USERNAME: ['username1', 'username2']
    #         },
    #     },
    #     'db': DB_INFO
    # }
    # req = get_validated_etl_request('prefeatures', etl_config, ETL_CONFIG_NAME_PREFEATURES_TEST)
    # # req = get_etl_req_prefeats(ETL_CONFIG_NAME_PREFEATURES_TEST, etl_config)
    #
    # # run test
    # setup_for_prefeatures_tests(req) # setup for test
    # etl_prefeatures_main(req) # run pipeline
    # verify_prefeatures_tests(req) # verify results


