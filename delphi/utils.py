import hashlib
import multiprocessing as mp
import sys
import threading
from functools import wraps
from queue import Queue
from typing import List, Union, Iterable, Any, TypeVar

import grpc
import numpy as np
from logzero import logger


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_example_key(content) -> str:
    return hashlib.sha1(content).hexdigest() + '.jpg'


def get_weights(targets: List[int], num_classes=2) -> List[int]:
    class_weights = [0] * num_classes
    classes, counts = np.unique(targets, return_counts=True)
    for i in range(len(classes)):
        class_weights[classes[i]] = len(targets) / float(counts[i])

    logger.info('Class weights: {}'.format(class_weights))

    weight = [0] * len(targets)
    for idx, val in enumerate(targets):
        weight[idx] = class_weights[val]

    return weight


T = TypeVar('T')


def bounded_iter(iterable: Iterable[T], semaphore: threading.Semaphore) -> Iterable[T]:
    for item in iterable:
        semaphore.acquire()
        yield item


def to_iter(queue: Union[Queue, mp.Queue]) -> Iterable[Any]:
    def iterate():
        while True:
            example = queue.get()
            if example is None:
                break
            yield example

    return iterate()


def log_exceptions(func):
    @wraps(func)
    def func_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(e)
            raise e

    return func_wrapper


def log_exceptions_and_abort(func):
    @wraps(func)
    def func_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(e)
            args[-1].abort(grpc.StatusCode.INTERNAL, str(sys.exc_info()[0]))

    return func_wrapper
