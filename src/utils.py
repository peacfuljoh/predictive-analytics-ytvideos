import json
from typing import Union, List


def save_json(path: str,
              obj: Union[List[dict], dict]):
    with open(path, 'w') as fp:
        json.dump(obj, fp, indent=4)