"""Featurization ETL utils"""

from typing import Generator, Optional, List, Union, Dict
import tempfile

import pandas as pd
import gensim as gs
from gensim.corpora import Dictionary

from src.crawler.crawler.config import DB_INFO, DB_CONFIG, DB_MONGO_CONFIG
from src.crawler.crawler.constants import (FEATURES_VECTOR_COL, VOCAB_VOCABULARY_COL, VOCAB_TIMESTAMP_COL,
                                           VOCAB_ETL_CONFIG_COL, PREFEATURES_TOKENS_COL, FEATURES_ETL_CONFIG_COL)
from src.crawler.crawler.utils.mongodb_engine import get_mongodb_records_gen, MongoDBEngine
from src.crawler.crawler.utils.misc_utils import get_ts_now_str, is_list_of_strings, df_generator_wrapper
from src.etl_pipelines.etl_request import ETLRequest, req_to_etl_config_record


DB_FEATURES_NOSQL_DATABASE = DB_INFO['DB_FEATURES_NOSQL_DATABASE'] # NoSQL features
DB_FEATURES_NOSQL_COLLECTIONS = DB_INFO['DB_FEATURES_NOSQL_COLLECTIONS']


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
            if key == 'username':
                assert isinstance(val, str) or is_list_of_strings(val)
            else:
                raise NotImplementedError(f'Extract condition {key} is not available.')

        # validate other extract options
        if 'limit' in config_:
            assert isinstance(config_['limit'], int)
        else:
            config_['limit'] = None

        for key in ['etl_config_prefeatures']:
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
            if key == 'username':
                assert isinstance(val, str) or is_list_of_strings(val)
            else:
                raise NotImplementedError(f'Extract condition {key} is not available.')

        # validate other extract options
        for key in ['etl_config_prefeatures', 'etl_config_vocabulary']:
            assert key in config_ and isinstance(config_[key], str)





""" Prefeatures """
def etl_extract_prefeature_records(req: Union[ETLRequestVocabulary, ETLRequestFeatures],
                                   projection: Optional[dict] = None,
                                   distinct: Optional[dict] = None) \
        -> Generator[pd.DataFrame, None, None]:
    """Get a generator of prefeature records."""
    assert projection is None or distinct is None  # at most one can be specified

    filter = None if (distinct is not None) else req.get_extract()['filters']

    return get_mongodb_records_gen(
        DB_FEATURES_NOSQL_DATABASE,
        DB_FEATURES_NOSQL_COLLECTIONS['prefeatures'],
        DB_MONGO_CONFIG,
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
    while not (df := next(df_gen)).empty:
        for doc in df[PREFEATURES_TOKENS_COL].values:
            yield doc.split(' ')
    return StopIteration

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

def convert_gs_dictionary_to_string(dictionary: Dictionary) -> str:
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
    # save config
    engine = MongoDBEngine(DB_MONGO_CONFIG,
                           database=DB_FEATURES_NOSQL_DATABASE,
                           collection=DB_FEATURES_NOSQL_COLLECTIONS['etl_config_vocabulary'],
                           verbose=True)
    d_req = req_to_etl_config_record(req, 'subset')
    engine.insert_one(d_req)

    # convert Gensim corpus to string
    vocab_txt: str = convert_gs_dictionary_to_string(dictionary)

    # save vocab
    rec_vocab = {
        VOCAB_VOCABULARY_COL: vocab_txt,
        VOCAB_TIMESTAMP_COL: get_ts_now_str('ms'),
        VOCAB_ETL_CONFIG_COL: req.name
    }

    engine = MongoDBEngine(DB_MONGO_CONFIG,
                           database=DB_FEATURES_NOSQL_DATABASE,
                           collection=DB_FEATURES_NOSQL_COLLECTIONS['vocabulary'],
                           verbose=True)
    engine.insert_one(rec_vocab)




""" Tokens """
def etl_load_vocab(req: ETLRequestFeatures) -> Dictionary:
    engine = MongoDBEngine(DB_MONGO_CONFIG,
                           database=DB_FEATURES_NOSQL_DATABASE,
                           collection=DB_FEATURES_NOSQL_COLLECTIONS['vocabulary'],
                           verbose=True)

    etl_config_name: str = req.get_extract()['etl_config_vocabulary']
    filter = dict(etl_config=etl_config_name)
    rec = engine.find_one(filter=filter)
    assert isinstance(rec, dict) and 'vocabulary' in rec
    dictionary = convert_string_to_gs_dictionary(rec['vocabulary'])

    return dictionary

@df_generator_wrapper
def etl_featurize_records_with_vocab(df_gen: Generator[pd.DataFrame, None, None],
                                     vocabulary: Dictionary,
                                     req: ETLRequestFeatures) \
        -> Generator[pd.DataFrame, None, None]:
    """Featurize prefeature records using vocabulary"""
    df = next(df_gen)
    if df.empty:
        raise StopIteration

    # map tokens to vectors in-place and rename col
    df[PREFEATURES_TOKENS_COL] = {i: vocabulary.doc2bow(rec[PREFEATURES_TOKENS_COL].split())
                                  for i, rec in df.iterrows()}
    df = df.rename(columns={PREFEATURES_TOKENS_COL: FEATURES_VECTOR_COL})

    # Note: df only contains etl_config_prefeatures column at this point

    return df

def etl_load_prefeatures_prepare_for_insert(df: pd.DataFrame,
                                            req: ETLRequestFeatures) \
        -> List[dict]:
    """
    Convert DataFrame with features info into dict for MongoDB insertion.
    """
    records_all: List[dict] = df.to_dict('records')
    for rec in records_all:
        # prefeatures_etl_config is already present, need to add etl configs for vocab and features stages
        rec[VOCAB_ETL_CONFIG_COL] = req.get_extract()[VOCAB_ETL_CONFIG_COL]
        rec[FEATURES_ETL_CONFIG_COL] = req.get_extract()[FEATURES_ETL_CONFIG_COL]

    return records_all

def etl_load_features_to_db(feat_gen: Generator[pd.DataFrame, None, None],
                            req: ETLRequestFeatures):
    """Save features config and feature records to database"""
    # save config
    engine = MongoDBEngine(DB_MONGO_CONFIG,
                           database=DB_FEATURES_NOSQL_DATABASE,
                           collection=DB_FEATURES_NOSQL_COLLECTIONS['etl_config_features'],
                           verbose=True)
    d_req = req_to_etl_config_record(req, 'subset')
    engine.insert_one(d_req)

    # save vocab
    engine = MongoDBEngine(DB_MONGO_CONFIG,
                           database=DB_FEATURES_NOSQL_DATABASE,
                           collection=DB_FEATURES_NOSQL_COLLECTIONS['features'],
                           verbose=True)
    while not (df := next(feat_gen)).empty:
        records: List[dict] = etl_load_prefeatures_prepare_for_insert(df, req)
        engine.insert_many(records)