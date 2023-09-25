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

from typing import Dict, Union
from PIL import Image

import pandas as pd

from src.preprocessor.featurization_etl_utils import ETLRequest, etl_extract_tabular, etl_extract_nontabular, \
    etl_clean_raw_data, etl_featurize


def etl_main(req: ETLRequest):
    """Entry point for ETL preprocessor"""
    data = etl_extract(req)
    etl_transform(data, req)
    return data
    etl_load(data, req)


""" Extract """
def etl_extract(req: ETLRequest) -> Dict[str, Union[pd.DataFrame, Dict[str, Image]]]:
    """Extract step of ETL pipeline"""
    df, info_tabular_extract = etl_extract_tabular(req)
    records = etl_extract_nontabular(df, info_tabular_extract)
    return dict(stats=df, images=records)


""" Transform """
def etl_transform(data: dict,
                  req: ETLRequest):
    """Transform step of ETL pipeline"""
    etl_clean_raw_data(data, req)
    etl_featurize(data, req)


""" Load """
def etl_load(data: dict,
             req: ETLRequest):
    """Load extracted features to feature store"""
    pass