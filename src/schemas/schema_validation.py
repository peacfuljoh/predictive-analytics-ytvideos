"""Utils for schema validation"""

from typing import List, Union


def validate_mongodb_records_schema_one(rec: dict, schema_info: dict) -> bool:
    """Validate schema for single MongoDB record pre-write"""
    if set(rec) != set(schema_info):
        return False
    for key, field_info in schema_info.items():
        if not isinstance(rec[key], field_info['type']):
            return False
    return True

def validate_mongodb_records_schema(recs: Union[dict, List[dict]], schema_info: dict) -> bool:
    """Validate that records follow the excepted schema, e.g. before writing records to a MongoDB collection."""
    if isinstance(recs, dict):
        return validate_mongodb_records_schema_one(recs, schema_info)
    for rec in recs:
        if not validate_mongodb_records_schema_one(rec, schema_info):
            return False
    return True
