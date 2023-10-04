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

from src.etl_pipelines.featurization_etl_utils import \
    ETLRequestFeatures, ETLRequestVocabulary, etl_extract_prefeature_records, etl_create_vocab, etl_load_vocab_to_db, \
    etl_featurize_records_with_vocab, etl_load_features_to_db, etl_load_vocab


def etl_features_main(req_vocab: ETLRequestVocabulary,
                      req_features: ETLRequestFeatures):
    """Entry point for ETL preprocessor"""
    etl_vocabulary_pipeline(req_vocab)
    etl_features_pipeline(req_features)

def etl_vocabulary_pipeline(req_vocab: ETLRequestVocabulary):
    df_gen = etl_extract_prefeature_records(req_vocab, distinct='tokens')
    vocabulary = etl_create_vocab(df_gen, req_vocab)
    etl_load_vocab_to_db(vocabulary, req_vocab)

def etl_features_pipeline(req_features: ETLRequestFeatures):
    df_gen = etl_extract_prefeature_records(req_features)
    vocabulary = etl_load_vocab(req_features)
    feat_gen = etl_featurize_records_with_vocab(df_gen, vocabulary, req_features)
    etl_load_features_to_db(feat_gen, req_features)
