
from typing import Union, Optional, Callable
import requests

from src.crawler.crawler.constants import ETL_CONFIG_VALID_KEYS_VOCAB, ETL_CONFIG_EXCLUDE_KEYS_VOCAB, \
    ETL_CONFIG_VALID_KEYS_FEATURES, ETL_CONFIG_EXCLUDE_KEYS_FEATURES, ETL_CONFIG_VALID_KEYS_PREFEATURES, \
    ETL_CONFIG_EXCLUDE_KEYS_PREFEATURES
from src.etl.etl_request import (ETLRequestPrefeatures, ETLRequestVocabulary, ETLRequestFeatures,
                                 ETLRequest, req_to_etl_config_record)
from src.crawler.crawler.config import CONFIGS_VALIDATE_ENDPOINT

# from src.ml.ml_request import MLRequest


def get_validated_etl_request(mode: str,
                              etl_config: dict,
                              etl_config_name: str = 'test1234567890',
                              validation_func: Optional[Callable] = None) \
        -> Union[ETLRequestPrefeatures, ETLRequestVocabulary, ETLRequestFeatures]:
    """Prepare ETL request objects with validation."""
    assert mode in ['prefeatures', 'vocabulary', 'features']#, 'models']
    if mode == 'prefeatures':
        req = ETLRequestPrefeatures(etl_config,
                                    etl_config_name,
                                    ETL_CONFIG_VALID_KEYS_PREFEATURES,
                                    ETL_CONFIG_EXCLUDE_KEYS_PREFEATURES)
    if mode == 'vocabulary':
        req = ETLRequestVocabulary(etl_config,
                                   etl_config_name,
                                   ETL_CONFIG_VALID_KEYS_VOCAB,
                                   ETL_CONFIG_EXCLUDE_KEYS_VOCAB)
    if mode == 'features':
        req = ETLRequestFeatures(etl_config,
                                 etl_config_name,
                                 ETL_CONFIG_VALID_KEYS_FEATURES,
                                 ETL_CONFIG_EXCLUDE_KEYS_FEATURES)
    # if mode == 'models':
    #     req = ETLRequestFeatures(etl_config,
    #                              etl_config_name,
    #                              ETL_CONFIG_VALID_KEYS_FEATURES,
    #                              ETL_CONFIG_EXCLUDE_KEYS_FEATURES)
    if validation_func is None:
        validate_etl_config_via_api(req, mode)
    else:
        validation_func(req, mode)
    return req

def validate_etl_config_via_api(req: ETLRequest,
                                collection: str):
    """Check that specified ETL config doesn't conflict with any existing configs in the ETL config db"""
    data = dict(
        config=req_to_etl_config_record(req, 'subset'),
        collection=collection
    )
    res = requests.post(CONFIGS_VALIDATE_ENDPOINT, json=data)
    is_valid = res.json()['valid']
    if not is_valid:
        raise Exception(f'The specified ETL pipeline options do not match those of '
                        f'the existing config for name {req.name}.')
    req.set_valid(True)
