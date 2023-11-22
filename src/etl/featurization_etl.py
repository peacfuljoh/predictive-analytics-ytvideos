"""
ETL (prefeatures -> vocabulary)
    - extract: load from prefeature database
    - transform: learn vocabulary
    - load: save vocabulary

ETL (prefeatures + vocabulary -> features)
    - extract: load from prefeature and vocabulary databases
    - transform: featurize tokens
    - load: send features to feature store
"""

from gensim.corpora import Dictionary

from src.etl.featurization_etl_utils import (ETLRequestFeatures, ETLRequestVocabulary,
                                             etl_create_vocab,etl_featurize_records_with_vocab,
                                             etl_extract_prefeature_records_ws, etl_load_vocab_to_db_ws,
                                             etl_load_vocab_from_db_ws, etl_load_features_to_db_ws)
from src.crawler.crawler.constants import PREFEATURES_ETL_CONFIG_COL, PREFEATURES_TOKENS_COL


def etl_features_main(req_vocab: ETLRequestVocabulary,
                      req_features: ETLRequestFeatures):
    """Entry point for ETL preprocessor"""
    etl_vocabulary_pipeline(req_vocab)
    etl_features_pipeline(req_features)

def etl_vocabulary_pipeline(req_vocab: ETLRequestVocabulary):
    # extract
    filter = {'$match': {PREFEATURES_ETL_CONFIG_COL: req_vocab.get_preconfig()[PREFEATURES_ETL_CONFIG_COL]}}
    distinct = dict(group=PREFEATURES_TOKENS_COL, filter=filter) # filter is applied first
    df_gen = etl_extract_prefeature_records_ws(req_vocab, distinct=distinct)

    # transform
    vocabulary: Dictionary = etl_create_vocab(df_gen, req_vocab)

    # load
    etl_load_vocab_to_db_ws(vocabulary, req_vocab)

def etl_features_pipeline(req_features: ETLRequestFeatures):
    # extract
    df_gen = etl_extract_prefeature_records_ws(req_features)
    vocabulary: dict = etl_load_vocab_from_db_ws(req_features)

    # transform
    feat_gen = etl_featurize_records_with_vocab(df_gen, vocabulary, req_features)

    # load
    etl_load_features_to_db_ws(feat_gen, req_features)
