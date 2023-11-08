
from typing import Union

from src.crawler.crawler.constants import ETL_CONFIG_VALID_KEYS_VOCAB, ETL_CONFIG_EXCLUDE_KEYS_VOCAB, \
    ETL_CONFIG_VALID_KEYS_FEATURES, ETL_CONFIG_EXCLUDE_KEYS_FEATURES, ETL_CONFIG_VALID_KEYS_PREFEATURES, \
    ETL_CONFIG_EXCLUDE_KEYS_PREFEATURES
from src.etl.etl_request import validate_etl_config
from src.etl.featurization_etl_utils import ETLRequestVocabulary, ETLRequestFeatures
from src.etl.prefeaturization_etl_utils import ETLRequestPrefeatures


def get_validated_etl_request(mode: str,
                              etl_config: dict,
                              etl_config_name: str = 'test1234567890') \
        -> Union[ETLRequestPrefeatures, ETLRequestVocabulary, ETLRequestFeatures]:
    """Prepare ETL request objects with validation."""
    assert mode in ['prefeatures', 'vocabulary', 'features']
    if mode == 'prefeatures':
        req = ETLRequestPrefeatures(etl_config,
                                    etl_config_name,
                                    ETL_CONFIG_VALID_KEYS_PREFEATURES,
                                    ETL_CONFIG_EXCLUDE_KEYS_PREFEATURES)
        db_ = req._db
        validate_etl_config(req,
                            db_['db_mongo_config'],
                            db_['db_info']['DB_FEATURES_NOSQL_DATABASE'],
                            db_['db_info']['DB_FEATURES_NOSQL_COLLECTIONS']['etl_config_prefeatures'])
    if mode == 'vocabulary':
        req = ETLRequestVocabulary(etl_config,
                                   etl_config_name,
                                   ETL_CONFIG_VALID_KEYS_VOCAB,
                                   ETL_CONFIG_EXCLUDE_KEYS_VOCAB)
        db_ = req._db
        validate_etl_config(req,
                            db_['db_mongo_config'],
                            db_['db_info']['DB_FEATURES_NOSQL_DATABASE'],
                            db_['db_info']['DB_FEATURES_NOSQL_COLLECTIONS']['etl_config_vocabulary'])
    if mode == 'features':
        req = ETLRequestFeatures(etl_config,
                                 etl_config_name,
                                 ETL_CONFIG_VALID_KEYS_FEATURES,
                                 ETL_CONFIG_EXCLUDE_KEYS_FEATURES)
        db_ = req._db
        validate_etl_config(req,
                            db_['db_mongo_config'],
                            db_['db_info']['DB_FEATURES_NOSQL_DATABASE'],
                            db_['db_info']['DB_FEATURES_NOSQL_COLLECTIONS']['etl_config_features'])
    return req
