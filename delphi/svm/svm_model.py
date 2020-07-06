import glob
import multiprocessing as mp
import os
import pickle
import queue
import threading
from typing import Callable, Iterator, List, Tuple, Any, Union

import torch
from logzero import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.svm import LinearSVC, SVC

from delphi.model import Model
from delphi.proto.learning_module_pb2 import InferRequest, InferResult
from delphi.svm.feature_provider import FeatureProvider, BATCH_SIZE, get_worker_feature_provider, \
    set_worker_feature_provider
from delphi.utils import log_exceptions


# return object_id, whether to preprocess, vector or (image, key)
@log_exceptions
def load_from_path(image_path: str) -> Tuple[str, bool, Union[List[float], Any]]:
    split = image_path.split('/')
    label = split[-2]
    name = split[-1]
    object_id = '{}/{}'.format(label, name)

    key = get_worker_feature_provider().get_result_key_name(name)
    cached_vector = get_worker_feature_provider().get_cached_vector(key)
    if cached_vector is not None:
        return object_id, False, cached_vector
    else:
        with open(image_path, 'rb') as f:
            content = f.read()

        return object_id, True, (get_worker_feature_provider().preprocess(content).numpy(), key)


# return object_id, whether to preprocess, vector or (image, key)
@log_exceptions
def load_from_content(request: InferRequest) -> Tuple[str, bool, Union[List[float], Any]]:
    key = get_worker_feature_provider().get_result_key_content(request.content)
    cached_vector = get_worker_feature_provider().get_cached_vector(key)
    if cached_vector is not None:
        return request.objectId, False, cached_vector
    else:
        return request.objectId, True, (get_worker_feature_provider().preprocess(request.content).numpy(), key)


class SVMModel(Model):

    def __init__(self, svc: Union[LinearSVC, SVC, CalibratedClassifierCV, VotingClassifier], version: int,
                 feature_provider: FeatureProvider, probability: bool):
        self._svc = svc
        self._version = version
        self._feature_provider = feature_provider
        self._probability = probability
        self._system_examples = []
        self._system_examples_lock = threading.Lock()

    @property
    def version(self) -> int:
        return self._version

    def infer(self, requests: Iterator[InferRequest]) -> Iterator[InferResult]:
        with mp.Pool(min(16, mp.cpu_count()), initializer=set_worker_feature_provider,
                     initargs=(self._feature_provider.feature_extractor,
                               self._feature_provider.cache)) as pool:
            images = pool.imap_unordered(load_from_content, requests, chunksize=64)
            yield from self._infer_inner(images)

    def infer_dir(self, directory: str, callback_fn: Callable[[int, float], None]) -> None:
        with mp.Pool(min(16, mp.cpu_count()), initializer=set_worker_feature_provider,
                     initargs=(self._feature_provider.feature_extractor,
                               self._feature_provider.cache)) as pool:
            images = pool.imap_unordered(load_from_path, glob.iglob(os.path.join(directory, '*/*')), chunksize=64)

            results = self._infer_inner(images)

            i = 0
            for result in results:
                i += 1
                # TODO(hturki): Should we get the target label in a less hacky way?
                callback_fn(int(result.objectId.split('/')[-2]), result.score)
                if i % 1000 == 0:
                    logger.info('{} examples scored so far'.format(i))

    def get_bytes(self) -> bytes:
        return pickle.dumps(self._svc)

    @property
    def scores_are_probabilities(self) -> bool:
        return self._probability

    def _infer_inner(self, images: Iterator[Tuple[str, bool, Union[List[float], Any]]]) -> Iterator[InferResult]:
        feature_queue = queue.Queue()

        @log_exceptions
        def process_uncached():
            cached = 0
            uncached = 0
            batch = []
            for object_id, should_process, payload in images:
                if should_process:
                    image, key = payload
                    batch.append((object_id,
                                  torch.from_numpy(image).to(self._feature_provider.device, non_blocking=True),
                                  key))
                    if len(batch) == BATCH_SIZE:
                        self._process_batch(batch, feature_queue)
                        batch = []
                    uncached += 1
                else:
                    feature_queue.put((object_id, payload))
                    cached += 1

            if len(batch) > 0:
                self._process_batch(batch, feature_queue)

            logger.info('{} cached examples, {} new examples preprocessed'.format(cached, uncached))
            feature_queue.put(None)

        threading.Thread(target=process_uncached, name='process-uncached').start()

        scored = 0
        queue_finished = False
        while not queue_finished:
            object_ids = []
            features = []

            item = feature_queue.get()
            if item is None:
                break

            object_ids.append(item[0])
            features.append(item[1])

            while True:
                try:
                    item = feature_queue.get(block=False)
                    if item is None:
                        queue_finished = True
                        break

                    object_ids.append(item[0])
                    features.append(item[1])
                except queue.Empty:
                    break

            if len(features) == 0:
                continue

            scores = self._svc.predict_proba(features) if self._probability else self._svc.decision_function(
                features)
            scored += len(object_ids)
            for i in range(len(object_ids)):
                if self._probability:
                    score = scores[i][1]
                    label = '1' if score >= 0.5 else '0'
                else:
                    score = scores[i]
                    label = '1' if score > 0 else '0'

                yield InferResult(objectId=object_ids[i], label=label, score=score, modelVersion=self.version)

        logger.info('{} examples scored'.format(scored))

    # label, image, key -> label, vector
    def _process_batch(self, items: List[Tuple[str, torch.Tensor, str]], feature_queue: queue.Queue) -> None:
        keys = [i[2] for i in items]
        tensor = torch.stack([i[1] for i in items])
        results = self._feature_provider.cache_and_get(keys, tensor, True)
        for item in items:
            feature_queue.put((item[0], results[item[2]]))
