"""
ETL (prefeatures -> features)
    - extract: load from prefeature database
    - transform: learn vocabulary and featurize tokens
    - load: send features to feature store
"""

from typing import Generator, Optional, List

import pandas as pd

from src.etl_pipelines.featurization_etl_utils import \
    ETLRequestFeatures, etl_extract_prefeature_records, etl_create_vocab, etl_load_vocab_to_db, \
    etl_featurize_records_with_vocab, etl_load_features_to_db



def etl_features_main(req: ETLRequestFeatures):
    """Entry point for ETL preprocessor"""
    # vocabulary
    df_gen = etl_features_vocab_extract(req)
    data = etl_features_vocab_transform(df_gen, req)
    etl_features_vocab_load(data, req)

    # tokens
    df_gen = etl_features_tokens_extract(req)
    feat_gen = etl_features_tokens_transform(df_gen, req)
    etl_features_load(feat_gen, req)


""" Vocabulary """
def etl_features_vocab_extract(req: ETLRequestFeatures) \
        -> Generator[pd.DataFrame, None, None]:
    """Extract step of ETL pipeline"""
    return etl_extract_prefeature_records(req, distinct='tokens')

def etl_features_vocab_transform(lst_gen: Generator[List[str], None, None],
                                 req: ETLRequestFeatures):
    """Transform step of ETL pipeline"""
    return etl_create_vocab(lst_gen, req)

def etl_features_vocab_load(data,
                            req: ETLRequestFeatures):
    """Load extracted features to feature store"""
    etl_load_vocab_to_db(data, req)


""" Tokens """
def etl_features_tokens_extract(req: ETLRequestFeatures) \
        -> Generator[pd.DataFrame, None, None]:
    """Extract step of ETL pipeline"""
    return etl_extract_prefeature_records(req)

def etl_features_tokens_transform(df_gen: Generator[pd.DataFrame, None, None],
                                  req: ETLRequestFeatures) \
        -> Generator[pd.DataFrame, None, None]:
    """Transform step of ETL pipeline"""
    return etl_featurize_records_with_vocab(df_gen, req)

def etl_features_load(feat_gen: Generator[pd.DataFrame, None, None],
                      req: ETLRequestFeatures):
    """Load extracted features to feature store"""
    etl_load_features_to_db(feat_gen, req)

