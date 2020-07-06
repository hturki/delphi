import os
import queue
import threading
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Iterator, Callable, Optional, Tuple

from logzero import logger

from delphi.context.data_manager_context import DataManagerContext
from delphi.model_trainer import DataRequirement
from delphi.proto.internal_pb2 import ExampleMetadata, GetExamplesRequest, GetExampleRequest
from delphi.proto.learning_module_pb2 import LabeledExample, ExampleSet
from delphi.utils import get_example_key, to_iter

TMP_DIR = 'test-0'

IGNORE_FILE = 'ignore'
TEST_RATIO = 0.2  # Hold out 20% of labeled examples as test


class DataManager(object):

    def __init__(self, context: DataManagerContext):
        self._context = context

        self._staging_dir = os.path.join(self._context.data_dir, 'examples-staging')
        os.makedirs(self._staging_dir, exist_ok=True)
        self._staging_lock = threading.Lock()

        self._examples_dir = os.path.join(self._context.data_dir, 'examples')
        for example_set in ExampleSet.keys():
            os.makedirs(os.path.join(self._examples_dir, example_set.lower()), exist_ok=True)

        self._examples_lock = threading.Lock()

        self._tmp_dir = os.path.join(self._examples_dir, TMP_DIR)
        os.makedirs(self._tmp_dir, exist_ok=True)

        self._example_counts = defaultdict(int)

        self._stored_examples_event = threading.Event()
        threading.Thread(target=self._promote_staging_examples, name='promote-staging-examples').start()

    def add_labeled_examples(self, examples: Iterator[LabeledExample]) -> None:
        data_requirement = self._get_data_requirement()

        if data_requirement is DataRequirement.MASTER_ONLY:
            if self._context.node_index == 0:
                self._store_labeled_examples(examples, None)
            else:
                self._context.nodes[0].api.AddLabeledExamples(examples)
        else:
            if self._context.node_index != 0:
                example_queue = queue.Queue()
                future = self._context.nodes[0].api.AddLabeledExamples.future(to_iter(example_queue))

                if data_requirement is DataRequirement.DISTRIBUTED_FULL:
                    self._store_labeled_examples(examples, example_queue.put)
                else:
                    def add_example(example: LabeledExample) -> None:
                        if example.exampleSet.value is ExampleSet.TEST or example.label == '1':
                            example_queue.put(example)

                    self._store_labeled_examples(examples, add_example)

                example_queue.put(None)
                future.result()
            else:
                self._store_labeled_examples(examples, None)

    @contextmanager
    def get_examples(self, example_set: ExampleSet) -> Iterator[str]:
        with self._examples_lock:
            if self._context.node_index != 0:
                if example_set is ExampleSet.TRAIN:
                    assert self._get_data_requirement() is not DataRequirement.MASTER_ONLY

                self._sync_with_master(example_set)
                yield os.path.join(self._examples_dir, self._to_dir(example_set))
            else:
                if example_set is ExampleSet.TEST:
                    for label in os.listdir(os.path.join(self._examples_dir, self._to_dir(example_set))):
                        os.makedirs(os.path.join(self._tmp_dir, label), exist_ok=True)
                        test_files = os.listdir(os.path.join(self._examples_dir, self._to_dir(example_set), label))
                        for i in range(0, len(test_files), len(self._context.nodes)):
                            os.rename(os.path.join(self._examples_dir, self._to_dir(example_set), label, test_files[i]),
                                      os.path.join(self._tmp_dir, label, test_files[i]))

                    yield os.path.join(self._tmp_dir)

                    for label in os.listdir(self._tmp_dir):
                        for tmp_file in os.listdir(os.path.join(self._tmp_dir, label)):
                            os.rename(os.path.join(self._tmp_dir, label, tmp_file),
                                      os.path.join(self._examples_dir, self._to_dir(example_set), label, tmp_file))
                else:
                    yield os.path.join(self._examples_dir, self._to_dir(example_set))

    def get_example_stream(self, example_set: ExampleSet, node_index: int) -> Iterator[ExampleMetadata]:
        assert self._examples_lock.locked()
        assert self._context.node_index == 0
        assert node_index != 0

        if example_set is ExampleSet.TRAIN:
            assert self._get_data_requirement() is not DataRequirement.MASTER_ONLY
            for label in os.listdir(os.path.join(self._examples_dir, self._to_dir(example_set))):
                if self._get_data_requirement() is DataRequirement.DISTRIBUTED_POSITIVES and label != '1':
                    continue

                for example in os.listdir(os.path.join(self._examples_dir, self._to_dir(example_set), label)):
                    yield ExampleMetadata(label=label, key=example)
        elif example_set is ExampleSet.TEST:
            for label in os.listdir(os.path.join(self._examples_dir, self._to_dir(example_set))):
                label_examples = os.listdir(os.path.join(self._examples_dir, self._to_dir(example_set), label))
                for i in range(node_index - 1, len(label_examples), len(self._context.nodes) - 1):
                    yield ExampleMetadata(label=label, key=label_examples[i])
        else:
            raise NotImplementedError('Unrecognized example set ' + self._to_dir(example_set))

    def get_example_location(self, example_set: ExampleSet, label: str, example: str) -> str:
        assert self._examples_lock.locked()
        assert self._context.node_index == 0

        return os.path.join(self._examples_dir, self._to_dir(example_set), label, example)

    def reset(self, train_only: bool):
        with self._staging_lock:
            self._clear_dir(self._staging_dir, train_only)

        with self._examples_lock:
            self._clear_dir(self._examples_dir, train_only)

    def _clear_dir(self, dir_path: str, train_only: bool):
        for child in os.listdir(dir_path):
            child_path = os.path.join(dir_path, child)
            if os.path.isdir(child_path):
                if child != 'test' or not train_only:
                    self._clear_dir(child_path, train_only)
            else:
                os.remove(child_path)

    def _store_labeled_examples(self, examples: Iterator[LabeledExample],
                                callback: Optional[Callable[[LabeledExample], None]]) -> None:
        with self._staging_lock:
            old_dirs = []
            for dir in os.listdir(self._staging_dir):
                for file in os.listdir(os.path.join(self._staging_dir, dir)):
                    if file != IGNORE_FILE:
                        old_dirs.extend(os.path.join(self._staging_dir, dir, file))

            for example in examples:
                example_file = get_example_key(example.content)
                self._remove_old_paths(example_file, old_dirs)

                if example.label != '-1':
                    if example.HasField('exampleSet'):
                        example_subdir = self._to_dir(example.exampleSet.value)
                    else:
                        example_subdir = 'unspecified'

                    os.makedirs(os.path.join(self._staging_dir, example_subdir, example.label), exist_ok=True)
                    example_path = os.path.join(self._staging_dir, example_subdir, example.label, example_file)
                    with open(example_path, 'wb') as f:
                        f.write(example.content)
                    logger.info('Saved example with label {} to path {}'.format(example.label, example_path))
                else:
                    logger.info('Example set to ignore - skipping')
                    with open(os.path.join(self._staging_dir, IGNORE_FILE), 'a+') as ignore_file:
                        ignore_file.write(example_file + '\n')

                if callback is not None:
                    callback(example)

        self._stored_examples_event.set()

    def _sync_with_master(self, example_set: ExampleSet) -> None:
        to_delete = defaultdict(set)
        for label in os.listdir(os.path.join(self._examples_dir, self._to_dir(example_set))):
            if self._get_data_requirement() is DataRequirement.DISTRIBUTED_POSITIVES \
                    and example_set is ExampleSet.TRAIN \
                    and label != '1':
                continue

            to_delete[label] = set(os.listdir(os.path.join(self._examples_dir, self._to_dir(example_set), label)))

        for example in self._context.nodes[0].internal.GetExamples(
                GetExamplesRequest(exampleSet=example_set, nodeIndex=self._context.node_index)):
            if example.key in to_delete[example.label]:
                to_delete[example.label].remove(example.key)
            else:
                example_content = self._context.nodes[0].internal.GetExample(
                    GetExampleRequest(exampleSet=example_set, label=example.label, key=example.key))
                label_dir = os.path.join(self._examples_dir, self._to_dir(example_set), example.label)
                os.makedirs(label_dir, exist_ok=True)
                example_path = os.path.join(label_dir, example.key)
                with open(example_path, 'wb') as file:
                    file.write(example_content.value)

        for label in to_delete:
            for file in to_delete[label]:
                os.remove(os.path.join(self._examples_dir, self._to_dir(example_set), label, file))

    def _promote_staging_examples(self):
        while True:
            try:
                self._stored_examples_event.wait()
                self._stored_examples_event.clear()

                new_positives = 0
                new_negatives = 0
                with self._examples_lock:
                    set_dirs = {}
                    for example_set in [ExampleSet.TRAIN, ExampleSet.TEST]:
                        old_dirs = []
                        for subdir in os.listdir(os.path.join(self._examples_dir, self._to_dir(example_set))):
                            old_dirs.extend(os.path.join(self._examples_dir, self._to_dir(example_set), subdir))
                        set_dirs[example_set] = old_dirs

                    with self._staging_lock:
                        for file in os.listdir(self._staging_dir):
                            if file == IGNORE_FILE:
                                with open(os.path.join(self._staging_dir, file)) as ignore_file:
                                    for line in ignore_file:
                                        for example_set in set_dirs:
                                            if self._remove_old_paths(line, set_dirs[example_set]):
                                                self._example_counts[example_set] -= 1
                            else:
                                dir_positives, dir_negatives = self._promote_staging_examples_dir(file, set_dirs)
                                new_positives += dir_positives
                                new_negatives += dir_negatives

                self._context.new_examples_callback(new_positives, new_negatives)
            except Exception as e:
                logger.exception(e)

    def _promote_staging_examples_dir(self, subdir: str, set_dirs: Dict['ExampleSet', List[str]]) -> Tuple[int, int]:
        assert subdir == self._to_dir(ExampleSet.TRAIN) \
               or subdir == self._to_dir(ExampleSet.TEST) \
               or subdir == 'unspecified'

        new_positives = 0
        new_negatives = 0
        for label in os.listdir(os.path.join(self._staging_dir, subdir)):
            staging_path = os.path.join(self._staging_dir, subdir, label)
            example_files = os.listdir(staging_path)
            if label == '1':
                new_positives += len(example_files)
            else:
                new_negatives += len(example_files)

            for example_file in example_files:
                for example_set in set_dirs:
                    if self._remove_old_paths(example_file, set_dirs[example_set]):
                        self._example_counts[example_set] -= 1

                if subdir == 'test' or (
                        subdir == 'unspecified' and self._example_counts[ExampleSet.TEST] <
                        self._example_counts[ExampleSet.TRAIN] * TEST_RATIO):
                    example_set = ExampleSet.TEST
                else:
                    example_set = ExampleSet.TRAIN

                self._example_counts[example_set] += 1
                os.makedirs(os.path.join(self._examples_dir, self._to_dir(example_set), label), exist_ok=True)
                example_path = os.path.join(self._examples_dir, self._to_dir(example_set), label, example_file)
                os.rename(os.path.join(staging_path, example_file), example_path)
                logger.info('Promoted example with label {} to path {}'.format(label, example_path))

        return new_positives, new_negatives

    def _get_data_requirement(self) -> DataRequirement:
        return max([x.data_requirement for x in self._context.get_active_trainers()], key=lambda y: y.value)

    @staticmethod
    def _remove_old_paths(example_file, old_dirs: List[str]) -> bool:
        for old_path in old_dirs:
            old_example_path = os.path.join(old_path, example_file)
            if os.path.exists(old_example_path):
                os.remove(old_example_path)
                logger.info('Removed old path {} for example'.format(old_example_path))
                return True

        return False

    @staticmethod
    def _to_dir(example_set: ExampleSet):
        return ExampleSet.Name(example_set).lower()
