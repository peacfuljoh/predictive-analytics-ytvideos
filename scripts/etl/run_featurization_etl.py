"""Script for experimenting with ETL pipeline"""

import copy

from src.crawler.crawler.config import DB_MONGO_CONFIG
from src.etl_pipelines.featurization_etl import etl_features_main
from src.etl_pipelines.featurization_etl_utils import (ETLRequestFeatures, ETLRequestVocabulary,
                                                       DB_FEATURES_NOSQL_DATABASE, DB_FEATURES_NOSQL_COLLECTIONS)
from src.crawler.crawler.constants import PREFEATURES_ETL_CONFIG_COL, VOCAB_ETL_CONFIG_COL
from src.etl_pipelines.etl_request import validate_etl_config


ETL_CONFIG_VOCAB_VALID_KEYS = dict(
    extract=['filters', 'etl_config_prefeatures'],
    transform=[],
    load=[],
    preconfig=[PREFEATURES_ETL_CONFIG_COL]
)
ETL_CONFIG_VOCAB_EXCLUDE_KEYS = dict(
    extract=[],
    transform=[],
    load=[],
    preconfig=[]
)

ETL_CONFIG_FEATURES_VALID_KEYS = dict(
    extract=['filters', 'etl_config_prefeatures', 'etl_config_vocab_name'],
    transform=[],
    load=[],
    preconfig=[PREFEATURES_ETL_CONFIG_COL, VOCAB_ETL_CONFIG_COL]
)
ETL_CONFIG_FEATURES_EXCLUDE_KEYS = dict(
    extract=[],
    transform=[],
    load=[],
    preconfig=[]
)


# set up etl config options
etl_config_prefeatures_name = 'test3'
etl_config_name = 'test217' # vocab and features


if etl_config_name == 'test217':
    etl_config_vocab = {
        'extract': {
            'filters': {
                'username': ['CNN', "TheYoungTurks", "FoxNews", "WashingtonPost", "msnbc", "NBCNews"]
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
                                 ETL_CONFIG_VOCAB_VALID_KEYS,
                                 ETL_CONFIG_VOCAB_EXCLUDE_KEYS)
validate_etl_config(req_vocab,
                    DB_MONGO_CONFIG,
                    DB_FEATURES_NOSQL_DATABASE,
                    DB_FEATURES_NOSQL_COLLECTIONS['etl_config_vocabulary'])

req_features = ETLRequestFeatures(etl_config_features,
                                  etl_config_name,
                                  ETL_CONFIG_FEATURES_VALID_KEYS,
                                  ETL_CONFIG_FEATURES_EXCLUDE_KEYS)
validate_etl_config(req_features,
                    DB_MONGO_CONFIG,
                    DB_FEATURES_NOSQL_DATABASE,
                    DB_FEATURES_NOSQL_COLLECTIONS['etl_config_features'])


# run pipeline
etl_features_main(req_vocab, req_features)
