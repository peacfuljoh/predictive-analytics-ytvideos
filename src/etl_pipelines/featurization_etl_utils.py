"""Featurization ETL utils"""

import copy

from src.crawler.crawler.config import DB_INFO, DB_CONFIG, DB_MONGO_CONFIG
from src.crawler.crawler.utils.mongodb_engine import MongoDBEngine, get_mongodb_records
from src.crawler.crawler.utils.misc_utils import is_datetime_formatted_str, df_generator_wrapper
from src.etl_pipelines.etl_request import ETLRequest


DB_FEATURES_NOSQL_DATABASE = DB_INFO['DB_FEATURES_NOSQL_DATABASE'] # NoSQL features
DB_FEATURES_NOSQL_COLLECTIONS = DB_INFO['DB_FEATURES_NOSQL_COLLECTIONS']


ETL_CONFIG_VALID_KEYS_FEATURES = dict(
    extract=['filters', 'limit'],
    transform=['include_additional_keys'],
    load=[]
)
ETL_CONFIG_EXCLUDE_FOR_FEATURES = dict(
    extract=['filters', 'limit'],
    transform=[],
    load=[]
)







""" ETL Request class for features processing """
class ETLRequestFeatures(ETLRequest):
    """
    Request object for Extract-Transform-Load operations.
    """
    def __init__(self,
                 config: dict,
                 name: str):
        super().__init__(config, name)

    def _validate_config_extract(self, config_: dict):
        # ensure specified options are a subset of valid options
        self._validate_config_keys(config_, 'extract')

        # validate extract filters
        for key, val in config_['filters'].items():
            if key == 'video_id':
                assert isinstance(val, str) or (isinstance(val, list) and all([isinstance(s, str) for s in val]))
            elif key == 'username':
                assert isinstance(val, str) or (isinstance(val, list) and all([isinstance(s, str) for s in val]))
            elif key == 'upload_date':
                fmt = '%Y-%m-%d'
                assert (is_datetime_formatted_str(val, fmt) or
                        (isinstance(val, list) and len(val) == 2 and all([is_datetime_formatted_str(s, fmt) for s in val])))
            elif key == 'timestamp_accessed':
                fmt = '%Y-%m-%d %H:%M:%S.%f'
                assert (is_datetime_formatted_str(val, fmt) or
                        (isinstance(val, list) and len(val) == 2 and all([is_datetime_formatted_str(s, fmt) for s in val])))
            else:
                raise NotImplementedError(f'Extract condition {key} is not available.')

        # validate other extract options
        if 'limit' in config_:
            assert isinstance(config_['limit'], int)
        else:
            config_['limit'] = None

    def _validate_config_transform(self, config_: dict):
        # ensure specified options are a subset of valid options
        self._validate_config_keys(config_, 'transform')

        # validate transform filters
        # ...

        # validate other transform options
        # ...

    def _validate_config_load(self, config_: dict):
        # ensure specified options are a subset of valid options
        self._validate_config_keys(config_, 'load')

        # validate load filters
        # ...

        # validate other load options
        # ...

    def _validate_config_keys(self,
                              sub_config: dict,
                              mode: str):
        """Validate keys specified in a sub-config."""
        assert mode in ['extract', 'transform', 'load']
        assert len(set(list(sub_config.keys())) - set(ETL_CONFIG_VALID_KEYS_FEATURES[mode])) == 0

    def get_config_as_dict(self,
                           mode: str = 'all',
                           validity_check: bool = True) \
            -> dict:
        """
        Return full config as a dictionary.

        Arg 'mode' determines what is included in returned dict:
        - 'all': include everything
        - 'prefeatures': only include fields relevant to prefeatures extract + transform (i.e. not filters or limit)
        """
        if validity_check:
            self._check_if_valid()

        assert mode in ['all', 'features']
        d = copy.deepcopy(dict(
            extract=self._extract,
            transform=self._transform,
            load=self._load
        ))
        if mode == 'all':
            pass
        elif mode == 'features':
            for key in d.keys():
                d[key] = {key_: val_ for key_, val_ in d[key].items()
                          if key_ not in ETL_CONFIG_EXCLUDE_FOR_FEATURES[key]}
        return {self.name: d}

def req_to_etl_config_record(req: ETLRequestFeatures) -> dict:
    """Get ETL config dict for insertion into prefeatures config db"""
    d_req = req.get_config_as_dict(mode='features', validity_check=False)
    d_req = {
        '_id': req.name,
        **d_req[req.name]
    }
    return d_req

def verify_valid_features_etl_config(req: ETLRequestFeatures):
    """Check that specified ETL config doesn't conflict with any existing configs in the ETL config db"""
    d_req = req_to_etl_config_record(req) # specified config
    engine = MongoDBEngine(DB_MONGO_CONFIG,
                           database=DB_FEATURES_NOSQL_DATABASE,
                           collection=DB_FEATURES_NOSQL_COLLECTIONS['etl_config_features'],
                           verbose=True)
    d_req_exist = engine.find_one(d_req['_id']) # existing config

    # check for invalid ETL config
    if (d_req_exist is not None) and (d_req_exist['_id'] == d_req['_id']):
        if d_req != d_req_exist: # ensure that options match exactly
            raise Exception(f'The specified ETL pipeline options do not match those of '
                            f'the existing config for name {req.name}.')

    # mark request as valid
    req.set_valid(True)