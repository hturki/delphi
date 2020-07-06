import os
import pickle
from typing import List, Optional, Dict

from logzero import logger

from delphi.svm.feature_cache import FeatureCache


class FSFeatureCache(FeatureCache):

    def __init__(self, feature_dir: str):
        self._feature_dir = feature_dir
        os.makedirs(feature_dir, exist_ok=True)

    def get(self, key: str) -> Optional[List[float]]:
        path = os.path.join(self._feature_dir, key)
        if not os.path.exists(path):
            return None

        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.exception(e)
            os.remove(path)
            return None

    def put(self, values: Dict[str, List[float]], expire: bool) -> None:
        for key in values:
            with open(os.path.join(self._feature_dir, key), 'wb') as f:
                pickle.dump(values[key], f)
