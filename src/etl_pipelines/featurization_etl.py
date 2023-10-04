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

from typing import Dict

from gensim.corpora import Dictionary

from src.etl_pipelines.featurization_etl_utils import \
    ETLRequestFeatures, ETLRequestVocabulary, etl_extract_prefeature_records, etl_create_vocab, etl_load_vocab_to_db, \
    etl_featurize_records_with_vocab, etl_load_features_to_db, etl_load_vocab
from src.crawler.crawler.constants import PREFEATURES_ETL_CONFIG_COL, PREFEATURES_TOKENS_COL, VOCAB_ETL_CONFIG_COL


def etl_features_main(req_vocab: ETLRequestVocabulary,
                      req_features: ETLRequestFeatures):
    """Entry point for ETL preprocessor"""
    etl_vocabulary_pipeline(req_vocab)
    etl_features_pipeline(req_features)

def etl_vocabulary_pipeline(req_vocab: ETLRequestVocabulary):
    # extract
    # TODO: update 'etl_config' in prefeatures records to 'etl_config_prefeatures'
    # TODO: add etl config names as fields in doc, not through _id
    # TODO: remove manual _id in prefeatures and other collections
    filter = {'$match': {PREFEATURES_ETL_CONFIG_COL: req_vocab.get_extract()['etl_config_prefeatures']}}
    df_gen = etl_extract_prefeature_records(req_vocab, distinct=dict(group=PREFEATURES_TOKENS_COL, filter=filter))

    # transform
    vocabulary: Dictionary = etl_create_vocab(df_gen, req_vocab)

    # load
    etl_load_vocab_to_db(vocabulary, req_vocab)

def etl_features_pipeline(req_features: ETLRequestFeatures):
    # extract
    df_gen = etl_extract_prefeature_records(req_features)
    vocabulary: Dictionary = etl_load_vocab(req_features)

    # transform
    feat_gen = etl_featurize_records_with_vocab(df_gen, vocabulary, req_features)

    # load
    etl_load_features_to_db(feat_gen, req_features)
