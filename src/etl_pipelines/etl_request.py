"""ETL request parent class"""

from typing import Union, Dict, List, Tuple
import copy

from src.crawler.crawler.utils.mongodb_engine import MongoDBEngine
from src.crawler.crawler.utils.misc_utils import is_subset, is_list_of_list_of_strings


class ETLRequest():
    """Parent class for ETL requests."""
    def __init__(self,
                 config: dict,
                 name: str,
                 valid_keys: Dict[str, List[str]],
                 exclude_keys: Dict[str, List[str]]):
        # fields
        self.name = name
        self._validate_key_dicts([valid_keys, exclude_keys])
        self._valid_keys = valid_keys
        self._exclude_keys = exclude_keys

        self._extract: dict = None
        self._transform: dict = None
        self._load: dict = None
        self._valid = False

        # set fields with validation
        config = self._validate_config(config)
        self._extract = config['extract']
        self._transform = config['transform']
        self._load = config['load']

    def _validate_key_dicts(self, dicts: List[dict]):
        """Check that key dicts are valid"""
        for d in dicts:
            assert (isinstance(d, dict) and is_list_of_list_of_strings(list(d.values())))

    def _validate_config(self, config: dict) -> dict:
        """Validate configuration"""
        for key in ['extract', 'transform', 'load']:
            if key not in config:
                config[key] = {}
        self._validate_config_extract(config['extract'])
        self._validate_config_transform(config['transform'])
        self._validate_config_load(config['load'])
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

    def _check_if_valid(self):
        """Check if config is valid"""
        if not self._valid:
            raise Exception('ETL request has not been validated.')

    def _validate_config_keys(self,
                              sub_config: dict,
                              mode: str):
        """Validate keys specified in a sub-config."""
        assert mode in ['extract', 'transform', 'load']
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
        """
        if validity_check:
            self._check_if_valid()

        assert mode in ['all', 'subset']
        d = copy.deepcopy(dict(
            extract=self._extract,
            transform=self._transform,
            load=self._load
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

def validate_etl_config(req: ETLRequest,
                        db_config: dict,
                        database: str,
                        collection: str):
    """Check that specified ETL config doesn't conflict with any existing configs in the ETL config db"""
    d_req = req_to_etl_config_record(req, 'subset') # specified config
    engine = MongoDBEngine(db_config,
                           database=database,
                           collection=collection,
                           verbose=True)
    d_req_exist = engine.find_one_by_id(d_req['_id']) # existing config

    # check for invalid ETL config
    if (d_req_exist is not None) and (d_req_exist['_id'] == d_req['_id']):
        if d_req != d_req_exist: # ensure that options match exactly
            raise Exception(f'The specified ETL pipeline options do not match those of '
                            f'the existing config for name {req.name}.')

    # mark request as valid
    req.set_valid(True)
