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
import numpy as np

from src.preprocessor.featurization_etl_utils import ETLRequest, etl_extract_tabular, etl_extract_nontabular, \
    replace_chars_in_str, etl_process_keywords, etl_process_tags


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
    return dict(
        stats=df,
        images=records
    )

""" Transform """
def etl_transform(data: dict,
                  req: ETLRequest):
    """Transform step of ETL pipeline"""
    etl_clean_raw_data(data, req)
    etl_featurize(data, req)

def etl_clean_raw_data(data: dict,
                       req: ETLRequest):
    """Clean the raw data"""
    # fill missing numerical values (zeroes)
    df = data['stats']
    username_video_id_pairs = df[['username', 'video_id']].drop_duplicates()
    for _, (username, video_id) in username_video_id_pairs.iterrows():
        mask = (df['username'] == username) * (df['video_id'] == video_id)
        df[mask] = df[mask].replace(0, np.nan).interpolate(method='linear', axis=0)

    # filter text fields: 'comment', 'title', 'keywords', 'description', 'tags'
    replace_chars_in_str(df, 'comment', charset='LNP')
    replace_chars_in_str(df, 'title', charset='LNP')
    etl_process_keywords(df)
    etl_process_tags(df)
    a = 5

    #TODO: clean up description

    # handle missing values


def etl_featurize(data: dict,
                  req: ETLRequest):
    """Map cleaned raw data to features"""
    pass # add features to data dict


""" Load """
def etl_load(data: dict,
             req: ETLRequest):
    """Load extracted features to feature store"""
    pass