"""
Preprocessor functionality:
- ETL (raw -> prefeatures)
    - extract: load from databases
    - transform: learn vocabulary and featurize tokens
    - load: send features to feature store
"""

from typing import Dict, Union, Generator
from PIL import Image

import pandas as pd

from src.etl_pipelines.featurization_etl_utils import ETLRequestFeatures


def etl_main(req: ETLRequestFeatures):
    """Entry point for ETL preprocessor"""
    df_gen = etl_extract(req)



""" Extract """
def etl_extract(req: ETLRequestFeatures) -> Dict[str, Union[Generator[pd.DataFrame, None, None], Dict[str, Image]]]:
    """Extract step of ETL pipeline"""
    pass


""" Transform """
def etl_transform(data: Dict[str, Union[Generator[pd.DataFrame, None, None], Dict[str, Image]]],
                  req: ETLRequestFeatures) \
        -> Dict[str, Generator[pd.DataFrame, None, None]]:
    """Transform step of ETL pipeline"""
    pass


""" Load """
def etl_load(data: Dict[str, Generator[pd.DataFrame, None, None]],
             req: ETLRequestFeatures):
    """Load extracted prefeatures to prefeature store"""
    pass
