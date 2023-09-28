"""
Preprocessor functionality:
- ETL (raw -> prefeatures)
    - extract: load from databases
    - transform: clean up raw data and extract text tokens
    - load: send prefeatures to prefeature store
"""

from typing import Dict, Union, Generator
from PIL import Image

import pandas as pd

from src.preprocessor.prefeaturization_etl_utils import ETLRequestPrefeatures, etl_extract_tabular, etl_extract_nontabular, \
    etl_clean_raw_data, etl_featurize, etl_load_prefeatures


def etl_main(req: ETLRequestPrefeatures,
             return_for_dashboard: bool = False):
    """Entry point for ETL preprocessor"""
    data = etl_extract(req)
    gen_raw_feats = etl_transform(data, req)
    if return_for_dashboard:
        return gen_raw_feats['stats']
    etl_load(gen_raw_feats, req)


""" Extract """
def etl_extract(req: ETLRequestPrefeatures) -> Dict[str, Union[Generator[pd.DataFrame, None, None], Dict[str, Image]]]:
    """Extract step of ETL pipeline"""
    df, info_tabular_extract = etl_extract_tabular(req)
    # records = etl_extract_nontabular(df, info_tabular_extract)
    records = None
    return dict(stats=df, images=records)


""" Transform """
def etl_transform(data: Dict[str, Union[Generator[pd.DataFrame, None, None], Dict[str, Image]]],
                  req: ETLRequestPrefeatures) \
        -> Dict[str, Generator[pd.DataFrame, None, None]]:
    """Transform step of ETL pipeline"""
    gen_stats = etl_clean_raw_data(data, req)
    gen_ims = None
    gen_raw_feats = etl_featurize({'stats': gen_stats, 'images': gen_ims}, req)
    return {'stats': gen_raw_feats}


""" Load """
def etl_load(data: Dict[str, Generator[pd.DataFrame, None, None]],
             req: ETLRequestPrefeatures):
    """Load extracted prefeatures to prefeature store"""
    etl_load_prefeatures(data, req)
