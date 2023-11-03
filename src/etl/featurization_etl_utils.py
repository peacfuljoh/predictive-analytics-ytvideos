"""Featurization ETL utils"""

from typing import Generator, Optional, List, Union, Tuple
import tempfile

import pandas as pd
import gensim as gs
from gensim.corpora import Dictionary

# from src.crawler.crawler.config import DB_INFO, DB_MONGO_CONFIG
from src.crawler.crawler.constants import (FEATURES_VECTOR_COL, VOCAB_VOCABULARY_COL, VOCAB_TIMESTAMP_COL,
                                           VOCAB_ETL_CONFIG_COL, PREFEATURES_TOKENS_COL, FEATURES_ETL_CONFIG_COL,
                                           PREFEATURES_ETL_CONFIG_COL, FEATURES_TIMESTAMP_COL, COL_USERNAME)
from db_engines.mongodb_engine import MongoDBEngine
from db_engines.mongodb_utils import get_mongodb_records_gen
from ytpa_utils.val_utils import is_list_of_strings
from ytpa_utils.time_utils import get_ts_now_str
from src.etl.etl_request import ETLRequest, req_to_etl_config_record
from src.crawler.crawler.utils.mongodb_utils_ytvideos import convert_ts_fmt_for_mongo_id
from src.schemas.schema_validation import validate_mongodb_records_schema
from src.schemas.schemas import SCHEMAS_MONGODB


# DB_FEATURES_NOSQL_DATABASE = DB_INFO['DB_FEATURES_NOSQL_DATABASE'] # NoSQL features
# DB_FEATURES_NOSQL_COLLECTIONS = DB_INFO['DB_FEATURES_NOSQL_COLLECTIONS']


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
        for key, val in config_['filters'].items():
            if key == COL_USERNAME:
                assert isinstance(val, str) or is_list_of_strings(val)
            elif key == PREFEATURES_ETL_CONFIG_COL:
                assert isinstance(val, str)
            else:
                raise NotImplementedError(f'Extract condition {key} is not available.')

    def _validate_config_preconfig(self, config_: dict):
        # ensure specified options are a subset of valid options
        self._validate_config_keys(config_, 'preconfig')

        # validate entries
        for key in [PREFEATURES_ETL_CONFIG_COL, VOCAB_ETL_CONFIG_COL]:
            assert key in config_ and isinstance(config_[key], str)





""" Prefeatures """
def etl_extract_prefeature_records(req: Union[ETLRequestVocabulary, ETLRequestFeatures],
                                   projection: Optional[dict] = None,
                                   distinct: Optional[dict] = None) \
        -> Generator[pd.DataFrame, None, None]:
    """
    Get a generator of prefeature records.

    If using 'distinct' input arg, filter is specified through the distinct dict:
        e.g. distinct = dict(group=<str>, filter=<dict>)
    If not using 'distinct' input arg, filter is taken from extraction config filters in ETL request
    """
    assert projection is None or distinct is None  # at most one can be specified

    filter = None if (distinct is not None) else req.get_extract()['filters']

    db_ = req.get_db()

    return get_mongodb_records_gen(
        # DB_FEATURES_NOSQL_DATABASE,
        # DB_FEATURES_NOSQL_COLLECTIONS['prefeatures'],
        # DB_MONGO_CONFIG,
        db_['db_info']['DB_FEATURES_NOSQL_DATABASE'],
        db_['db_info']['DB_FEATURES_NOSQL_COLLECTIONS']['prefeatures'],
        db_['db_mongo_config'],
        filter=filter,
        projection=projection,
        distinct=distinct
    )


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
    # get all unique token strings
    # tokens_all: List[str] = []
    # while not (df := next(df_gen)).empty:
    #     tokens_all += list(df[PREFEATURES_TOKENS_COL])

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

def convert_gs_dictionary_to_string(dictionary: Dictionary) -> bytes:
    """Convert corpora to string"""
    with tempfile.NamedTemporaryFile() as fp:
        dictionary.save_as_text(fp.file.name) # , sort_by_word=True) # save to text
        return fp.read()

def convert_string_to_gs_dictionary(s: str) -> Dictionary:
    """Convert pre-formatted string to corpora"""
    with tempfile.NamedTemporaryFile() as fp:
        fp.write(s)
        return Dictionary.load_from_text(fp.file.name)

def etl_load_vocab_to_db(dictionary: Dictionary,
                         req: ETLRequestVocabulary):
    """Load vocabulary into vocab store"""
    db_ = req.get_db()
    mongo_config = db_['db_mongo_config']
    database = db_['db_info']['DB_FEATURES_NOSQL_DATABASE']
    collection_config = db_['db_info']['DB_FEATURES_NOSQL_COLLECTIONS']['etl_config_vocabulary']
    collection_vocab = db_['db_info']['DB_FEATURES_NOSQL_COLLECTIONS']['vocabulary']

    # save config
    engine = MongoDBEngine(mongo_config, database=database, collection=collection_config, verbose=True)
    d_req = req_to_etl_config_record(req, 'subset')
    engine.insert_one(d_req)

    # convert Gensim corpus to string
    vocab_txt: bytes = convert_gs_dictionary_to_string(dictionary)

    # save vocab
    rec_vocab = {
        VOCAB_VOCABULARY_COL: vocab_txt,
        VOCAB_TIMESTAMP_COL: get_ts_now_str('ms'),
        VOCAB_ETL_CONFIG_COL: req.name
    }
    assert validate_mongodb_records_schema(rec_vocab, SCHEMAS_MONGODB['vocabulary'])

    engine = MongoDBEngine(mongo_config, database=database, collection=collection_vocab, verbose=True)
    engine.insert_one(rec_vocab)




""" Tokens """
def etl_load_vocab_from_db(req: ETLRequestFeatures,
                           timestamp_vocab: Optional[str] = None) \
        -> dict:
    """Load vocabulary given specified options"""
    db_ = req.get_db()
    mongo_config = db_['db_mongo_config']
    database = db_['db_info']['DB_FEATURES_NOSQL_DATABASE']
    collection_vocab = db_['db_info']['DB_FEATURES_NOSQL_COLLECTIONS']['vocabulary']

    engine = MongoDBEngine(mongo_config, database=database, collection=collection_vocab, verbose=True)

    # TODO: replace with aggregation pipeline (first filter, then pick record with max timestamp_vocabulary)
    # get records
    etl_config_name = req.get_preconfig()[VOCAB_ETL_CONFIG_COL]
    filter = {VOCAB_ETL_CONFIG_COL: etl_config_name}
    if timestamp_vocab is not None:
        filter[VOCAB_TIMESTAMP_COL] = timestamp_vocab
    recs_all = engine.find_many_gen(filter=filter)

    # pick most recent
    dfs = [df for df in recs_all]
    df = pd.concat(dfs, axis=0, ignore_index=True)
    rec: pd.DataFrame = df.loc[df[VOCAB_TIMESTAMP_COL] == df[VOCAB_TIMESTAMP_COL].max()] # argmax not allowed for strings
    assert len(rec) == 1
    rec = rec.iloc[0].to_dict()

    # convert vocab string to Dictionary object
    assert VOCAB_VOCABULARY_COL in rec
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

def etl_load_features_to_db(feat_gen: Generator[pd.DataFrame, None, None],
                            req: ETLRequestFeatures):
    """Save features config and feature records to database"""
    db_ = req.get_db()
    mongo_config = db_['db_mongo_config']
    database = db_['db_info']['DB_FEATURES_NOSQL_DATABASE']
    collection_config = db_['db_info']['DB_FEATURES_NOSQL_COLLECTIONS']['etl_config_features']
    collection_features = db_['db_info']['DB_FEATURES_NOSQL_COLLECTIONS']['features']

    # save config
    engine = MongoDBEngine(mongo_config, database=database, collection=collection_config, verbose=True)
    d_req = req_to_etl_config_record(req, 'subset')
    engine.insert_one(d_req)

    # save vocab
    ts_feat = get_ts_now_str('ms') # has to be same for all records in this run
    engine = MongoDBEngine(mongo_config, database=database, collection=collection_features, verbose=True)
    for df in feat_gen:
        records: List[dict] = etl_load_prefeatures_prepare_for_insert(df, ts_feat, req)
        print(f"etl_load_features_to_db() -> Inserting {len(df)} records in collection {collection_features} "
              f"of database {database}")
        engine.insert_many(records)



