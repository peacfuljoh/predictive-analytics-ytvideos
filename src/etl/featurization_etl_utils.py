"""Featurization ETL utils"""

from typing import Generator, Optional, List, Union
import tempfile
import requests

import pandas as pd
import gensim as gs
from gensim.corpora import Dictionary

# from src.crawler.crawler.config import DB_INFO, DB_MONGO_CONFIG
from src.crawler.crawler.constants import (FEATURES_VECTOR_COL, VOCAB_VOCABULARY_COL, VOCAB_TIMESTAMP_COL,
                                           VOCAB_ETL_CONFIG_COL, PREFEATURES_TOKENS_COL, FEATURES_ETL_CONFIG_COL,
                                           PREFEATURES_ETL_CONFIG_COL, FEATURES_TIMESTAMP_COL, COL_USERNAME,
                                           TIMESTAMP_CONVERSION_FMTS_DECODE, TIMESTAMP_CONVERSION_FMTS_ENCODE,
                                           COL_TIMESTAMP_ACCESSED)
from db_engines.mongodb_engine import MongoDBEngine
from db_engines.mongodb_utils import get_mongodb_records_gen
from ytpa_utils.val_utils import is_list_of_strings
from ytpa_utils.time_utils import get_ts_now_str
from ytpa_utils.df_utils import df_dt_codec
from src.etl.etl_request import ETLRequest, req_to_etl_config_record
from src.crawler.crawler.utils.mongodb_utils_ytvideos import convert_ts_fmt_for_mongo_id, post_one_record
from src.schemas.schema_validation import validate_mongodb_records_schema
from src.schemas.schemas import SCHEMAS_MONGODB

from ytpa_api_utils.websocket_utils import df_generator_ws
from ytpa_api_utils.request_utils import df_sender_for_insert
from src.crawler.crawler.config import PREFEATURES_ENDPOINT, VOCABULARY_ENDPOINT, MONGO_INSERT_MANY_ENDPOINT




STOPLIST = 'for a of the and to in on is this at as be it that by are was'
STOP_WORD_LIST = set(STOPLIST.split() + [''])





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





""" Prefeatures """
def etl_extract_prefeature_records_ws(req: Union[ETLRequestVocabulary, ETLRequestFeatures],
                                      projection: Optional[dict] = None,
                                      distinct: Optional[dict] = None) \
        -> Generator[pd.DataFrame, None, None]:
    """Websocket stream version of prefeature record generator"""
    assert projection is None or distinct is None  # at most one can be specified

    filter = None if (distinct is not None) else req.get_extract()['filters']

    extract_options = {'filter': filter, 'projection': projection, 'distinct': distinct}
    etl_config_options = {'extract': extract_options}
    df_gen = df_generator_ws(PREFEATURES_ENDPOINT, etl_config_options, transformations=TIMESTAMP_CONVERSION_FMTS_DECODE)
    return df_gen




""" Vocabulary """
def gen_docs(df_gen: Generator[pd.DataFrame, None, None]):
    """
    Generate docs one at a time from a DataFrame generator.
    Yields a list of strings one-by-one, then a StopIteration.
    """
    for df in df_gen:
        for doc in df[PREFEATURES_TOKENS_COL].values:
            yield doc.split(' ')

def etl_create_vocab(df_gen: Generator[pd.DataFrame, None, None],
                     req: ETLRequestVocabulary) \
        -> Dictionary:
    """Create vocabulary from tokens in prefeature records"""
    # create vocabulary
    dictionary = gs.corpora.Dictionary(gen_docs(df_gen))

    # remove stop words and words that appear only once
    stop_ids = [
        dictionary.token2id[stopword]
        for stopword in STOP_WORD_LIST
        if stopword in dictionary.token2id
    ]
    once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq == 1]
    dictionary.filter_tokens(stop_ids + once_ids)
    dictionary.compactify() # reset word ids

    return dictionary

def convert_gs_dictionary_to_string(dictionary: Dictionary) -> str:
    """Convert corpora to string"""
    with tempfile.NamedTemporaryFile() as fp:
        dictionary.save_as_text(fp.file.name) # , sort_by_word=True) # save to text
        return fp.read().decode()

def convert_string_to_gs_dictionary(s: str) -> Dictionary:
    """Convert string to corpora"""
    with tempfile.NamedTemporaryFile() as fp:
        fp.write(s.encode('utf-8'))
        return Dictionary.load_from_text(fp.file.name)

def make_and_validate_vocab_record(vocab_txt: str,
                                   req: ETLRequestVocabulary) -> dict:
    """..."""
    rec_vocab = {
        VOCAB_VOCABULARY_COL: vocab_txt,
        VOCAB_TIMESTAMP_COL: get_ts_now_str('ms'),
        VOCAB_ETL_CONFIG_COL: req.name
    }
    assert validate_mongodb_records_schema(rec_vocab, SCHEMAS_MONGODB['vocabulary'])
    return rec_vocab

def etl_load_vocab_to_db_ws(dictionary: Dictionary,
                            req: ETLRequestVocabulary):
    """Load vocabulary into vocab store"""
    # save config
    d_req = req_to_etl_config_record(req, 'subset')
    res = post_one_record('DB_FEATURES_NOSQL_DATABASE', 'etl_config_vocabulary', d_req)
    print(res.json())

    # convert Gensim corpus to string
    vocab_txt: str = convert_gs_dictionary_to_string(dictionary)

    # save vocab
    rec_vocab = make_and_validate_vocab_record(vocab_txt, req)
    res = post_one_record('DB_FEATURES_NOSQL_DATABASE', 'vocabulary', rec_vocab)
    print(res.json())







""" Tokens """
def etl_load_vocab_meta_from_db(engine: MongoDBEngine,
                                filter: Optional[dict] = None) \
        -> pd.DataFrame:
    """Load vocabulary info without vocab itself. Used to inspect metadata."""
    return engine.find_many(filter=filter, projection={VOCAB_VOCABULARY_COL: 0})

def etl_load_vocab_setup_engine(req: ETLRequestFeatures):
    """Setup MongoDB engine for loading vocabulary records."""
    db_ = req.get_db()
    mongo_config = db_['db_mongo_config']
    database = db_['db_info']['DB_FEATURES_NOSQL_DATABASE']
    collection_vocab = db_['db_info']['DB_FEATURES_NOSQL_COLLECTIONS']['vocabulary']

    return MongoDBEngine(mongo_config, database=database, collection=collection_vocab, verbose=True)

def etl_load_vocab_from_db(req: ETLRequestFeatures,
                           timestamp_vocab: Optional[str] = None,
                           as_gs_dict: bool = True) \
        -> dict:
    """Load vocabulary given specified options"""
    engine = etl_load_vocab_setup_engine(req)

    # get records (choose timestamp before loading entire vocabulary)
    etl_config_name = req.get_preconfig()[VOCAB_ETL_CONFIG_COL]
    filter = {VOCAB_ETL_CONFIG_COL: etl_config_name}
    filter[VOCAB_TIMESTAMP_COL] = timestamp_vocab if timestamp_vocab is not None else \
        etl_load_vocab_meta_from_db(engine, filter=filter)[VOCAB_TIMESTAMP_COL].max()
    df = engine.find_many(filter=filter)
    assert len(df) == 1
    rec: dict = df.iloc[0].to_dict()

    # convert vocab string to Dictionary object
    assert VOCAB_VOCABULARY_COL in rec
    if as_gs_dict:
        rec[VOCAB_VOCABULARY_COL] = convert_string_to_gs_dictionary(rec[VOCAB_VOCABULARY_COL])

    return rec

def etl_load_vocab_from_db_ws(req: ETLRequestFeatures,
                              timestamp_vocab: Optional[str] = None,
                              as_gs_dict: bool = True) \
        -> dict:
    """Load vocabulary given specified options"""
    msg = {
        VOCAB_ETL_CONFIG_COL: req.get_preconfig()[VOCAB_ETL_CONFIG_COL],
        VOCAB_TIMESTAMP_COL: timestamp_vocab
    }
    res = requests.post(VOCABULARY_ENDPOINT, json=msg)
    rec = res.json()

    if as_gs_dict:
        rec[VOCAB_VOCABULARY_COL] = convert_string_to_gs_dictionary(rec[VOCAB_VOCABULARY_COL])

    return rec

def etl_featurize_records_with_vocab(df_gen: Generator[pd.DataFrame, None, None],
                                     vocabulary: dict,
                                     req: ETLRequestFeatures) \
        -> Generator[pd.DataFrame, None, None]:
    """Featurize prefeature records using vocabulary"""
    for df in df_gen:
        # map tokens to vectors in-place and rename col
        df[PREFEATURES_TOKENS_COL] = {i: vocabulary[VOCAB_VOCABULARY_COL].doc2bow(rec[PREFEATURES_TOKENS_COL].split())
                                      for i, rec in df.iterrows()}
        df = df.rename(columns={PREFEATURES_TOKENS_COL: FEATURES_VECTOR_COL})

        # add vocabulary metadata
        df[VOCAB_ETL_CONFIG_COL] = req.get_preconfig()[VOCAB_ETL_CONFIG_COL]
        df[VOCAB_TIMESTAMP_COL] = vocabulary[VOCAB_TIMESTAMP_COL]

        # Note: df only contains etl_config_prefeatures column at this point
        yield df

def etl_load_prefeatures_prepare_for_insert(df: pd.DataFrame,
                                            ts_feat: str,
                                            req: ETLRequestFeatures) \
        -> List[dict]:
    """
    Convert DataFrame with features info into dict for MongoDB insertion.
    """
    records_all: List[dict] = df.to_dict('records')
    for rec in records_all:
        rec[FEATURES_ETL_CONFIG_COL] = req.name
        rec[FEATURES_TIMESTAMP_COL] = ts_feat
        rec['_id'] += '_' + convert_ts_fmt_for_mongo_id(ts_feat)[1]

    assert validate_mongodb_records_schema(records_all, SCHEMAS_MONGODB['features'])

    return records_all

def etl_load_features_to_db_ws(feat_gen: Generator[pd.DataFrame, None, None],
                               req: ETLRequestFeatures):
    """Save features config and feature records to database"""
    d_req = req_to_etl_config_record(req, 'subset')
    res = post_one_record('DB_FEATURES_NOSQL_DATABASE', 'etl_config_features', d_req)
    print(res.json())

    # insert records
    ts_feat = get_ts_now_str('ms') # has to be same for all records in this run
    def preprocess_func(df: pd.DataFrame) -> dict:
        df_dt_codec(df, {COL_TIMESTAMP_ACCESSED: TIMESTAMP_CONVERSION_FMTS_ENCODE[COL_TIMESTAMP_ACCESSED]})
        records = etl_load_prefeatures_prepare_for_insert(df, ts_feat, req)
        return {'database': 'DB_FEATURES_NOSQL_DATABASE', 'collection': 'features', 'records': records}

    df_sender_for_insert(MONGO_INSERT_MANY_ENDPOINT, preprocess_func, feat_gen, print_json=True)
