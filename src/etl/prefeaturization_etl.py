"""
ETL (raw -> prefeatures)
    - extract: load from raw data store
    - transform: clean up raw data and prefeaturize (extract text tokens)
    - load: send prefeatures to prefeature store
"""

from typing import Dict, Union, Generator
from PIL import Image

import pandas as pd

from src.etl.prefeaturization_etl_utils import (etl_extract_tabular_ws, etl_extract_nontabular,
                                                etl_clean_raw_data, etl_featurize, etl_load_prefeatures_ws)
from src.etl.etl_request import ETLRequestPrefeatures



def etl_prefeatures_main(req: ETLRequestPrefeatures,
                         return_for_dashboard: bool = False):
    """Entry point for ETL etl"""
    data = etl_prefeatures_extract(req)
    gen_raw_feats = etl_prefeatures_transform(data, req)
    if return_for_dashboard:
        return gen_raw_feats['stats']
    etl_prefeatures_load(gen_raw_feats, req)


""" Extract """
def etl_prefeatures_extract(req: ETLRequestPrefeatures) \
        -> Dict[str, Union[Generator[pd.DataFrame, None, None], Dict[str, Image]]]:
    """Extract step of ETL pipeline"""
    df_gen = etl_extract_tabular_ws(req) # raw data generator
    records = None #etl_extract_nontabular(df, info_tabular_extract)
    return dict(stats=df_gen, images=records)


""" Transform """
def etl_prefeatures_transform(data: Dict[str, Union[Generator[pd.DataFrame, None, None], Dict[str, Image]]],
                              req: ETLRequestPrefeatures) \
        -> Dict[str, Generator[pd.DataFrame, None, None]]:
    """Transform step of ETL pipeline"""
    gen_stats = etl_clean_raw_data(data, req)
    gen_ims = None
    gen_raw_feats = etl_featurize({'stats': gen_stats, 'images': gen_ims}, req)
    return {'stats': gen_raw_feats}


""" Load """
def etl_prefeatures_load(data: Dict[str, Generator[pd.DataFrame, None, None]],
                         req: ETLRequestPrefeatures):
    """Load extracted prefeatures to prefeature store"""
    etl_load_prefeatures_ws(data, req)
