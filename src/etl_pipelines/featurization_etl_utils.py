"""Featurization ETL utils"""

from typing import Generator, Optional, List, Union
import tempfile

import pandas as pd
import gensim as gs

from src.crawler.crawler.config import DB_INFO, DB_CONFIG, DB_MONGO_CONFIG
from src.crawler.crawler.utils.mongodb_engine import get_mongodb_record_gen_features, MongoDBEngine
from src.crawler.crawler.utils.misc_utils import get_ts_now_str
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
                assert isinstance(val, str) or (isinstance(val, list) and all([isinstance(s, str) for s in val]))
            else:
                raise NotImplementedError(f'Extract condition {key} is not available.')

        # validate other extract options
        assert 'etl_config_vocab_name' in config_

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
                assert isinstance(val, str) or (isinstance(val, list) and all([isinstance(s, str) for s in val]))
            else:
                raise NotImplementedError(f'Extract condition {key} is not available.')

        # validate other extract options
        if 'limit' in config_:
            assert isinstance(config_['limit'], int)
        else:
            config_['limit'] = None




""" Prefeatures """
def etl_extract_prefeature_records(req: Union[ETLRequestVocabulary, ETLRequestFeatures],
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
def gen_docs(df_gen: Generator[pd.DataFrame, None, None]):
    """Generate docs one at a time from a DataFrame generator."""
    while not (df := next(df_gen)).empty:
        for doc in df['tokens'].values:
            yield doc.split(' ')
    return StopIteration

def etl_create_vocab(df_gen: Generator[pd.DataFrame, None, None],
                     req: ETLRequestVocabulary) \
        -> gs.corpora:
    """Create vocabulary from tokens in prefeature records"""
    # get all unique token strings
    # tokens_all: List[str] = []
    # while not (df := next(df_gen)).empty:
    #     tokens_all += list(df['tokens'])

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

def convert_gensim_corpora_to_string(dictionary: gs.corpora) -> str:
    """Convert corpora to string"""
    # TODO: implement this with stream object
    with tempfile.NamedTemporaryFile() as fobj:
        dictionary.save_as_text(fobj.file.name)  # , sort_by_word=True) # save to text
        return fobj.read()

def etl_load_vocab_to_db(dictionary: gs.corpora,
                         req: ETLRequestVocabulary):
    """Load vocabulary into vocab store"""
    # save config
    engine = MongoDBEngine(DB_MONGO_CONFIG,
                           database=DB_FEATURES_NOSQL_DATABASE,
                           collection=DB_FEATURES_NOSQL_COLLECTIONS['etl_config_vocabulary'],
                           verbose=True)
    d_req = req_to_etl_config_record(req, 'subset')
    engine.insert_one(d_req)
    engine.update_one({'_id': 'test'}, {'$set': {'extract': d_req['extract']}}, upsert=True)

    # convert Gensim corpus to string
    vocab_txt: str = convert_gensim_corpora_to_string(dictionary)

    # save vocab
    rec_vocab = {
        # '_id': ??, # unnecessary
        'vocabulary': vocab_txt,
        'timestamp': get_ts_now_str('ms'),
        'etl_config': req.name
    }

    engine = MongoDBEngine(DB_MONGO_CONFIG,
                           database=DB_FEATURES_NOSQL_DATABASE,
                           collection=DB_FEATURES_NOSQL_COLLECTIONS['vocabulary'],
                           verbose=True)
    engine.insert_one(rec_vocab)




""" Tokens """
def etl_load_vocab(req: ETLRequestFeatures) -> gs.corpora:
    # TODO: next
    pass

def etl_featurize_records_with_vocab(df_gen: Generator[pd.DataFrame, None, None],
                                     vocabulary: gs.corpora,
                                     req: ETLRequestFeatures) \
        -> Generator[pd.DataFrame, None, None]:
    pass

def etl_load_features_to_db(feat_gen: Generator[pd.DataFrame, None, None],
                            req: ETLRequestFeatures):
    pass
