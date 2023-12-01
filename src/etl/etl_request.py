"""ETL request classes and helper methods"""

from typing import Dict, List
import copy

from ytpa_utils.val_utils import (is_subset, is_list_of_list_of_strings, is_list_of_strings, is_datetime_formatted_str,
                                  is_list_of_list_of_time_range_strings)

from src.crawler.crawler.constants import (COL_VIDEO_ID, COL_USERNAME, COL_UPLOAD_DATE, DATE_FMT,
                                           COL_TIMESTAMP_ACCESSED, TIMESTAMP_FMT, PREFEATURES_ETL_CONFIG_COL,
                                           VOCAB_ETL_CONFIG_COL)



class ETLRequest():
    """Parent class for ETL requests."""
    def __init__(self,
                 config: dict,
                 name: str,
                 valid_keys: Dict[str, List[str]],
                 exclude_keys: Dict[str, List[str]]):
        # fields
        self.name = name # mainly used for db insertions
        self._validate_key_dicts([valid_keys, exclude_keys])
        self._valid_keys = valid_keys
        self._exclude_keys = exclude_keys

        self._extract: dict = None
        self._transform: dict = None
        self._load: dict = None
        self._preconfig: dict = None
        self._db: dict = None
        self._valid = False
        self._config_keys = ['extract', 'transform', 'load', 'preconfig', 'db']

        # set fields with validation
        config = self._validate_config(config)
        self._extract = config['extract']
        self._transform = config['transform']
        self._load = config['load']
        self._preconfig = config['preconfig']
        self._db = config['db']

    def _validate_key_dicts(self, dicts: List[dict]):
        """Check that key dicts are valid"""
        for d in dicts:
            assert (isinstance(d, dict) and is_list_of_list_of_strings(list(d.values())))

    def _validate_config(self, config: dict) -> dict:
        """Validate configuration"""
        for key in self._config_keys:
            if key not in config:
                config[key] = {}
        self._validate_config_extract(config['extract'])
        self._validate_config_transform(config['transform'])
        self._validate_config_load(config['load'])
        self._validate_config_preconfig(config['preconfig'])
        self._validate_config_db(config['db'])
        return config

    def _validate_config_extract(self, config: dict):
        """Validate extract config"""
        self._validate_config_keys(config, 'extract')

        # ... other validation goes here ...

    def _validate_config_transform(self, config: dict):
        """Validate transform config"""
        self._validate_config_keys(config, 'transform')

        # ... other validation goes here ...

    def _validate_config_load(self, config: dict):
        """Validate load config"""
        self._validate_config_keys(config, 'load')

        # ... other validation goes here ...

    def _validate_config_preconfig(self, config: dict):
        """Validate preconfig"""
        self._validate_config_keys(config, 'preconfig')

    def _validate_config_db(self, config: dict):
        """Validate preconfig"""
        pass
        # self._validate_config_keys(config, 'db_info')

    def _check_if_valid(self):
        """Check if config is valid"""
        if not self._valid:
            raise Exception('ETL request has not been validated.')

    def _validate_config_keys(self,
                              sub_config: dict,
                              mode: str):
        """Validate keys specified in a sub-config."""
        assert mode in self._config_keys
        assert is_subset(sub_config, self._valid_keys[mode])

    def get_config_as_dict(self,
                           mode: str = 'all',
                           validity_check: bool = True) \
            -> dict:
        """
        Return config as a dictionary.

        Arg 'mode' determines what is included in returned dict:
        - 'all': include everything
        - 'subset': only include fields relevant to ETL steps (i.e. not filters or limit)

        Note: 'db' is automatically excluded.
        """
        if validity_check:
            self._check_if_valid()

        assert mode in ['all', 'subset']
        d = copy.deepcopy(dict(
            extract=self._extract,
            transform=self._transform,
            load=self._load,
            preconfig=self._preconfig
        ))
        if mode == 'all':
            pass
        elif mode == 'subset':
            for key in d.keys():
                d[key] = {key_: val_ for key_, val_ in d[key].items() if key_ not in self._exclude_keys[key]}
        return {self.name: d}

    def get_extract(self) -> dict:
        """Get extract configuration"""
        self._check_if_valid()
        return self._extract

    def get_transform(self) -> dict:
        """Get transform configuration"""
        self._check_if_valid()
        return self._transform

    def get_load(self) -> dict:
        """Get load configuration"""
        self._check_if_valid()
        return self._load

    def get_preconfig(self) -> dict:
        """Get preconfig"""
        self._check_if_valid()
        return self._preconfig

    def get_db(self) -> dict:
        """Get preconfig"""
        self._check_if_valid()
        return self._db

    def set_valid(self, valid: bool):
        """Set config validity"""
        self._valid = valid



def req_to_etl_config_record(req: ETLRequest,
                             mode: str) \
        -> dict:
    """Get ETL config dict for insertion into config db"""
    d_req = req.get_config_as_dict(mode=mode, validity_check=False)
    d_req = {
        '_id': req.name,
        **d_req[req.name]
    }
    return d_req





""" ETL request class for prefeatures processing """
class ETLRequestPrefeatures(ETLRequest):
    """
    Request object for Extract-Transform-Load operations.
    """
    def _validate_config_extract(self, config_: dict):
        # ensure specified options are a subset of valid options
        self._validate_config_keys(config_, 'extract')

        # validate extract filters
        if 'filters' not in config_:
            config_['filters'] = {}
        for key, val in config_['filters'].items():
            if key == COL_VIDEO_ID:
                assert isinstance(val, str) or is_list_of_strings(val)
            elif key == COL_USERNAME:
                assert isinstance(val, str) or is_list_of_strings(val)
            elif key == COL_UPLOAD_DATE:
                fmt = DATE_FMT
                func = lambda s: is_datetime_formatted_str(s, fmt)
                assert is_datetime_formatted_str(val, fmt) or is_list_of_list_of_time_range_strings(val, func, num_ranges=1)
            elif key == COL_TIMESTAMP_ACCESSED:
                fmt = TIMESTAMP_FMT
                func = lambda s: is_datetime_formatted_str(s, fmt)
                assert is_datetime_formatted_str(val, fmt) or is_list_of_list_of_time_range_strings(val, func, num_ranges=1)
            else:
                raise NotImplementedError(f'Extract condition {key} is not available.')

        # validate other extract options
        if 'limit' in config_:
            assert config_['limit'] is None or isinstance(config_['limit'], int)
        else:
            config_['limit'] = None

    def _validate_config_transform(self, config_: dict):
        # ensure specified options are a subset of valid options
        self._validate_config_keys(config_, 'transform')

        # validate transform filters
        # ...

        # validate other transform options
        if 'include_additional_keys' in config_:
            assert isinstance(config_['include_additional_keys'], list)
            assert is_list_of_strings(list(config_['include_additional_keys']))

    def _validate_config_db(self, config: dict):
        # config keys
        if 0:
            assert set(config) == {'db_info', 'db_mysql_config', 'db_mongo_config'}

            # db info (database/table/collection names, etc.)
            info = config['db_info']
            assert set(info) == {
                'DB_VIDEOS_DATABASE', 'DB_VIDEOS_TABLES',
                'DB_VIDEOS_NOSQL_DATABASE', 'DB_VIDEOS_NOSQL_COLLECTIONS',
                'DB_FEATURES_NOSQL_DATABASE', 'DB_FEATURES_NOSQL_COLLECTIONS',
                'DB_MODELS_NOSQL_DATABASE', 'DB_MODELS_NOSQL_COLLECTIONS'

            }
            assert set(info['DB_VIDEOS_TABLES']) == {'users', 'meta', 'stats'}
            assert set(info['DB_VIDEOS_NOSQL_COLLECTIONS']) == {'thumbnails'}
            assert set(info['DB_FEATURES_NOSQL_COLLECTIONS']) == {'prefeatures', 'features', 'vocabulary',
                                                                   'etl_config_prefeatures', 'etl_config_features',
                                                                   'etl_config_vocabulary'}
            assert set(info['DB_MODELS_NOSQL_COLLECTIONS']) == {'models', 'meta'}

            # db configs (server connection credentials)
            assert set(config['db_mysql_config']) == {'host', 'user', 'password'}
            assert set(config['db_mongo_config']) == {'host', 'port'}







""" ETL Request class for features processing """
class ETLRequestVocabulary(ETLRequest):
    """
    Request object for Extract-Transform-Load operations.
    """
    def _validate_config_extract(self, config_: dict):
        # ensure specified options are a subset of valid options
        self._validate_config_keys(config_, 'extract')

        # validate extract filters
        if 'filters' not in config_:
            config_['filters'] = {}
        for key, val in config_['filters'].items():
            if key == COL_USERNAME:
                assert isinstance(val, str) or is_list_of_strings(val)
            else:
                raise NotImplementedError(f'Extract condition {key} is not available.')

        # validate other extract options
        if 'limit' in config_:
            assert isinstance(config_['limit'], int)
        else:
            config_['limit'] = None

    def _validate_config_preconfig(self, config_: dict):
        # ensure specified options are a subset of valid options
        self._validate_config_keys(config_, 'preconfig')

        # validate entries
        for key in [PREFEATURES_ETL_CONFIG_COL]:
            assert key in config_ and isinstance(config_[key], str)


class ETLRequestFeatures(ETLRequest):
    """
    Request object for Extract-Transform-Load operations.
    """
    def _validate_config_extract(self, config_: dict):
        # ensure specified options are a subset of valid options
        self._validate_config_keys(config_, 'extract')

        # validate extract filters
        if 'filters' not in config_:
            config_['filters'] = {}
        for key, val in config_['filters'].items():
            if key == COL_USERNAME:
                assert isinstance(val, str) or is_list_of_strings(val)
            elif key == PREFEATURES_ETL_CONFIG_COL:
                assert isinstance(val, str)
            else:
                raise NotImplementedError(f'Extract condition {key} is not available.')

        # validate other extract options
        if 'limit' in config_:
            assert isinstance(config_['limit'], int)
        else:
            config_['limit'] = None

    def _validate_config_preconfig(self, config_: dict):
        # ensure specified options are a subset of valid options
        self._validate_config_keys(config_, 'preconfig')

        # validate entries
        for key in [PREFEATURES_ETL_CONFIG_COL, VOCAB_ETL_CONFIG_COL]:
            assert key in config_ and isinstance(config_[key], str)


