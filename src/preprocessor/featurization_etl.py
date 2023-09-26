"""
Preprocessor functionality:
- ETL
    - extract: load from databases
    - transform:
        - clean up raw data, apply filters
        - handle missing fields (impute, mark, invalidate)
        - featurize
    - load: send features to feature store
"""

from typing import Dict, Union, Generator
from PIL import Image

import pandas as pd

from src.preprocessor.featurization_etl_utils import ETLRequest, etl_extract_tabular, etl_extract_nontabular, \
    etl_clean_raw_data, etl_featurize


def etl_main(req: ETLRequest):
    """Entry point for ETL preprocessor"""
    data = etl_extract(req)
    gen_raw_feats = etl_transform(data, req)
    return gen_raw_feats
    etl_load(data, req)


""" Extract """
def etl_extract(req: ETLRequest) -> Dict[str, Union[Generator[pd.DataFrame, None, None], Dict[str, Image]]]:
    """Extract step of ETL pipeline"""
    df, info_tabular_extract = etl_extract_tabular(req)
    # records = etl_extract_nontabular(df, info_tabular_extract)
    return dict(stats=df)#, images=records)


""" Transform """
def etl_transform(data: Dict[str, Union[Generator[pd.DataFrame, None, None], Dict[str, Image]]],
                  req: ETLRequest):
    """Transform step of ETL pipeline"""
    gen_stats = etl_clean_raw_data(data, req)
    gen_raw_feats = etl_featurize({'stats': gen_stats}, req)
    return gen_raw_feats


""" Load """
def etl_load(data: dict,
             req: ETLRequest):
    """Load extracted features to feature store"""
    pass
