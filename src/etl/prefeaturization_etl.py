"""
ETL (raw -> prefeatures)
    - extract: load from databases
    - transform: clean up raw raw_data and extract text tokens
    - load: send prefeatures to prefeature store
"""

from typing import Dict, Union, Generator
from PIL import Image

import pandas as pd

from ytpa_api_utils.websocket_utils import df_generator_ws

from src.etl.prefeaturization_etl_utils import (ETLRequestPrefeatures, etl_extract_tabular,
                                                etl_extract_nontabular, etl_clean_raw_data, etl_featurize,
                                                etl_load_prefeatures)
from src.crawler.crawler.config import RAWDATA_JOIN_ENDPOINT
from src.crawler.crawler.constants import TIMESTAMP_CONVERSION_FMTS_DECODE


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
    """
    Extract step of ETL pipeline.

    Debug commands to verify equivalence of direct and API methods of generating from etl_extract_tabular():

    from src.crawler.crawler.constants import TIMESTAMP_CONVERSION_FMTS_DECODE; from ytpa_utils.df_utils import df_dt_codec
    df1 = pd.concat([df for df in df_gen], ignore_index=True) # df_gen from df_generator_ws()
    df_gen, info_tabular_extract, _ = etl_extract_tabular(req); df2 = pd.concat([df for df in df_gen])
    d1 = df1.sort_values(by=['timestamp_accessed']).reset_index(drop=True); d2 = df2.sort_values(by=['timestamp_accessed']).reset_index(drop=True)
    [d1.shape, d2.shape]
    d1.equals(d2)

    from ytpa_utils.misc_utils import print_df_full
    print_df_full(d1.iloc[:5]); print_df_full(d2.iloc[:5])
    """
    # local df_gen
    # df_gen, info_tabular_extract, _ = etl_extract_tabular(req)

    # df_gen through API (websocket)
    etl_config_options = {'name': req.name, 'extract': req.get_extract()}
    df_gen = df_generator_ws(RAWDATA_JOIN_ENDPOINT, etl_config_options, transformations=TIMESTAMP_CONVERSION_FMTS_DECODE)

    # df_ = next(df_gen)
    # from src.crawler.crawler.utils.misc_utils import print_df_full
    # print_df_full(df_)
    # data_send = df_.to_dict('records')

    # records = etl_extract_nontabular(df, info_tabular_extract)
    records = None
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
    etl_load_prefeatures(data, req)
