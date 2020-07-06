import threading

from delphi.model_trainer import ModelTrainer


class ModelTrainerBase(ModelTrainer):

    def __init__(self):
        self._latest_version = 0
        self._version_lock = threading.Lock()

    def get_new_version(self):
        with self._version_lock:
            self._latest_version += 1
            version = self._latest_version
        return version
