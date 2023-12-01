"""Script for experimenting with ETL pipeline"""

import copy

from src.etl.etl_request_utils import get_validated_etl_request
from src.etl.featurization_etl import etl_features_main
from src.crawler.crawler.constants import PREFEATURES_ETL_CONFIG_COL, VOCAB_ETL_CONFIG_COL, COL_USERNAME






# set up pipeline_scripts config options
etl_config_prefeatures_name = 'test3'
etl_config_name = 'test5544' # vocab and features

if etl_config_name == 'test5544':
    etl_config_vocab = {
        'extract': {
            'filters': {
                COL_USERNAME: ['CNN', "TheYoungTurks", "FoxNews", "WashingtonPost", "msnbc", "NBCNews"]
            },
        },
        'preconfig': {
            PREFEATURES_ETL_CONFIG_COL: etl_config_prefeatures_name
        }
    }

    etl_config_features = copy.deepcopy(etl_config_vocab)
    etl_config_features['preconfig'][VOCAB_ETL_CONFIG_COL] = etl_config_name
    etl_config_features['extract']['filters'][PREFEATURES_ETL_CONFIG_COL] = etl_config_prefeatures_name

# set up request objects
req_vocab = get_validated_etl_request('vocabulary', etl_config_vocab, etl_config_name)
req_features = get_validated_etl_request('features', etl_config_features, etl_config_name)

# run pipeline
etl_features_main(req_vocab, req_features)


