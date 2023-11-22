"""Schemas for databases that don't automatically enforce a schema (e.g. MongoDB)"""

from src.crawler.crawler.constants import (COL_USERNAME, COL_TIMESTAMP_ACCESSED,
                                           COL_VIDEO_ID, PREFEATURES_ETL_CONFIG_COL,
                                           PREFEATURES_TOKENS_COL, COL_LIKE_COUNT,
                                           COL_COMMENT_COUNT, COL_SUBSCRIBER_COUNT, COL_VIEW_COUNT,
                                           VOCAB_VOCABULARY_COL, VOCAB_TIMESTAMP_COL, VOCAB_ETL_CONFIG_COL,
                                           FEATURES_ETL_CONFIG_COL, FEATURES_VECTOR_COL,
                                           FEATURES_TIMESTAMP_COL, MODEL_MODEL_OBJ, MODEL_META_ID, MODEL_SPLIT_NAME,
                                           MODEL_DICT_PREPROCESSOR, MODEL_DICT_DATA_BOW, MODEL_DICT_MODEL,
                                           MODEL_DICT_CONFIG)


""" MongoDB schemas """
d_type_str = dict(type=str)
d_type_int = dict(type=int)
d_type_list = dict(type=list)
d_type_dict = dict(type=dict)

MONGO_REC_SCHEMA_PREFEATURES = {
    '_id': d_type_str,
    COL_USERNAME: d_type_str,
    COL_VIDEO_ID: d_type_str,
    COL_TIMESTAMP_ACCESSED: d_type_str,
    PREFEATURES_ETL_CONFIG_COL: d_type_str,
    COL_LIKE_COUNT: d_type_int,
    COL_COMMENT_COUNT: d_type_int,
    COL_SUBSCRIBER_COUNT: d_type_int,
    COL_VIEW_COUNT: d_type_int,
    PREFEATURES_TOKENS_COL: d_type_str
}
MONGO_REC_SCHEMA_VOCABULARY = {
    VOCAB_VOCABULARY_COL: d_type_str,
    VOCAB_TIMESTAMP_COL: d_type_str,
    VOCAB_ETL_CONFIG_COL: d_type_str
}
MONGO_REC_SCHEMA_FEATURES = {
    '_id': d_type_str,
    COL_USERNAME: d_type_str,
    COL_VIDEO_ID: d_type_str,
    COL_SUBSCRIBER_COUNT: d_type_int,
    COL_VIEW_COUNT: d_type_int,
    COL_LIKE_COUNT: d_type_int,
    COL_COMMENT_COUNT: d_type_int,
    FEATURES_VECTOR_COL: d_type_list,
    PREFEATURES_ETL_CONFIG_COL: d_type_str,
    VOCAB_ETL_CONFIG_COL: d_type_str,
    FEATURES_ETL_CONFIG_COL: d_type_str,
    COL_TIMESTAMP_ACCESSED: d_type_str,
    VOCAB_TIMESTAMP_COL: d_type_str,
    FEATURES_TIMESTAMP_COL: d_type_str,
}
MONGO_REC_SCHEMA_MODELS = {
    MODEL_MODEL_OBJ: d_type_dict,
    MODEL_META_ID: d_type_str,
    MODEL_SPLIT_NAME: d_type_str
}
MONGO_REC_SCHEMA_MODELS_PARAMS = {
    MODEL_DICT_PREPROCESSOR: d_type_dict,
    MODEL_DICT_DATA_BOW: d_type_list,
    MODEL_DICT_MODEL: d_type_dict,
    MODEL_DICT_CONFIG: d_type_dict
}

SCHEMAS_MONGODB = dict(
    prefeatures=MONGO_REC_SCHEMA_PREFEATURES,
    vocabulary=MONGO_REC_SCHEMA_VOCABULARY,
    features=MONGO_REC_SCHEMA_FEATURES,
    models=MONGO_REC_SCHEMA_MODELS,
    models_params=MONGO_REC_SCHEMA_MODELS_PARAMS
)
