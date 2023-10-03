"""Featurization ETL utils"""

from typing import Generator, Optional, List

import pandas as pd

from src.crawler.crawler.config import DB_INFO, DB_CONFIG, DB_MONGO_CONFIG
from src.crawler.crawler.utils.mongodb_engine import get_mongodb_record_gen_features
from src.etl_pipelines.etl_request import ETLRequest


DB_FEATURES_NOSQL_DATABASE = DB_INFO['DB_FEATURES_NOSQL_DATABASE'] # NoSQL features
DB_FEATURES_NOSQL_COLLECTIONS = DB_INFO['DB_FEATURES_NOSQL_COLLECTIONS']




""" ETL Request class for features processing """
class ETLRequestFeatures(ETLRequest):
    """
    Request object for Extract-Transform-Load operations.
    """
    def _validate_config_extract(self, config_: dict):
        # ensure specified options are a subset of valid options
        self._validate_config_keys(config_, 'extract')

        # validate extract filters
        for key, val in config_['filters'].items():
            # if key == 'video_id':
            #     assert isinstance(val, str) or (isinstance(val, list) and all([isinstance(s, str) for s in val]))
            if key == 'username':
                assert isinstance(val, str) or (isinstance(val, list) and all([isinstance(s, str) for s in val]))
            # elif key == 'upload_date':
            #     fmt = '%Y-%m-%d'
            #     assert (is_datetime_formatted_str(val, fmt) or
            #             (isinstance(val, list) and len(val) == 2 and all([is_datetime_formatted_str(s, fmt) for s in val])))
            # elif key == 'timestamp_accessed':
            #     fmt = '%Y-%m-%d %H:%M:%S.%f'
            #     assert (is_datetime_formatted_str(val, fmt) or
            #             (isinstance(val, list) and len(val) == 2 and all([is_datetime_formatted_str(s, fmt) for s in val])))
            else:
                raise NotImplementedError(f'Extract condition {key} is not available.')

        # validate other extract options
        if 'limit' in config_:
            assert isinstance(config_['limit'], int)
        else:
            config_['limit'] = None




""" Prefeatures """
def etl_extract_prefeature_records(req: ETLRequestFeatures,
                                   projection: Optional[dict] = None,
                                   distinct: Optional[str] = None) \
        -> Generator[pd.DataFrame, None, None]:
    """Get a generator of prefeature records."""
    assert (projection is None and distinct is not None) or (projection is not None and distinct is None)

    return get_mongodb_record_gen_features(
        DB_FEATURES_NOSQL_DATABASE,
        DB_FEATURES_NOSQL_COLLECTIONS['prefeatures'],
        DB_MONGO_CONFIG,
        req.get_extract()['filters'],
        projection=projection,
        distinct=distinct
    )


""" Vocabulary """
def etl_create_vocab(lst_gen: Generator[pd.DataFrame, None, None],
                     req: ETLRequestFeatures):
    """Create vocabulary from tokens in prefeature records"""
    # get all unique token strings
    tokens_all: List[str] = []
    while not (df := next(lst_gen)).empty:
        tokens_all += list(df['tokens'])

    # create vocabulary
    a = 5


def etl_load_vocab_to_db(data,
                         req: ETLRequestFeatures):
    pass



""" Tokens """
def etl_featurize_records_with_vocab(df_gen: Generator[pd.DataFrame, None, None],
                                     req: ETLRequestFeatures) \
        -> Generator[pd.DataFrame, None, None]:
    pass

def etl_load_features_to_db(feat_gen: Generator[pd.DataFrame, None, None],
                            req: ETLRequestFeatures):
    pass
