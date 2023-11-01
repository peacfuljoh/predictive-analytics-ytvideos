"""Script for experimenting with ETL pipeline"""

import copy

from src.crawler.crawler.config import DB_MONGO_CONFIG
from src.etl.featurization_etl import etl_features_main
from src.etl.featurization_etl_utils import (ETLRequestFeatures, ETLRequestVocabulary,
                                             DB_FEATURES_NOSQL_DATABASE, DB_FEATURES_NOSQL_COLLECTIONS)
from src.crawler.crawler.constants import (PREFEATURES_ETL_CONFIG_COL, VOCAB_ETL_CONFIG_COL, ETL_CONFIG_VALID_KEYS_VOCAB,
                                           ETL_CONFIG_EXCLUDE_KEYS_VOCAB, ETL_CONFIG_VALID_KEYS_FEATURES,
                                           ETL_CONFIG_EXCLUDE_KEYS_FEATURES, COL_USERNAME, FEATURES_ETL_CONFIG_COL)
from src.etl.etl_request import validate_etl_config


# set up pipeline_scripts config options
etl_config_prefeatures_name = 'test3'
etl_config_name = 'test21715' # vocab and features


if etl_config_name == 'test21715':
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
req_vocab = ETLRequestVocabulary(etl_config_vocab,
                                 etl_config_name,
                                 ETL_CONFIG_VALID_KEYS_VOCAB,
                                 ETL_CONFIG_EXCLUDE_KEYS_VOCAB)
validate_etl_config(req_vocab,
                    DB_MONGO_CONFIG,
                    DB_FEATURES_NOSQL_DATABASE,
                    DB_FEATURES_NOSQL_COLLECTIONS['etl_config_vocabulary'])

req_features = ETLRequestFeatures(etl_config_features,
                                  etl_config_name,
                                  ETL_CONFIG_VALID_KEYS_FEATURES,
                                  ETL_CONFIG_EXCLUDE_KEYS_FEATURES)
validate_etl_config(req_features,
                    DB_MONGO_CONFIG,
                    DB_FEATURES_NOSQL_DATABASE,
                    DB_FEATURES_NOSQL_COLLECTIONS['etl_config_features'])


# run pipeline
etl_features_main(req_vocab, req_features)

