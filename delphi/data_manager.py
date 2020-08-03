import queue
import threading
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Iterable, Callable, Optional, Tuple

from logzero import logger

from delphi.context.data_manager_context import DataManagerContext
from delphi.model_trainer import DataRequirement
from delphi.proto.internal_pb2 import ExampleMetadata, GetExamplesRequest, GetExampleRequest
from delphi.proto.learning_module_pb2 import LabeledExample, ExampleSet, LabeledExampleRequest
from delphi.utils import get_example_key, to_iter

TMP_DIR = 'test-0'

IGNORE_FILE = 'ignore'
TEST_RATIO = 0.2  # Hold out 20% of labeled examples as test


class DataManager(object):

    def __init__(self, context: DataManagerContext):
        self._context = context

        self._staging_dir = self._context.data_dir / 'examples-staging'
        self._staging_dir.mkdir(parents=True, exist_ok=True)
        self._staging_lock = threading.Lock()

        self._examples_dir = self._context.data_dir / 'examples'
        for example_set in ExampleSet.keys():
            example_dir = self._examples_dir / example_set.lower()
            example_dir.mkdir(parents=True, exist_ok=True)

        self._examples_lock = threading.Lock()

        self._tmp_dir = self._examples_dir / TMP_DIR
        self._tmp_dir.mkdir(parents=True, exist_ok=True)

        self._example_counts = defaultdict(int)

        self._stored_examples_event = threading.Event()
        threading.Thread(target=self._promote_staging_examples, name='promote-staging-examples').start()

    def add_labeled_examples(self, examples: Iterable[LabeledExample]) -> None:
        data_requirement = self._get_data_requirement()

        if data_requirement is DataRequirement.MASTER_ONLY:
            if self._context.node_index == 0:
                self._store_labeled_examples(examples, None)
            else:
                example_queue = queue.Queue()
                future = self._context.nodes[0].api.AddLabeledExamples.future(to_iter(example_queue))
                example_queue.put(LabeledExampleRequest(searchId=self._context.search_id))
                for example in examples:
                    example_queue.put(LabeledExampleRequest(example=example))

                example_queue.put(None)
                future.result()
        else:
            if self._context.node_index != 0:
                example_queue = queue.Queue()
                future = self._context.nodes[0].api.AddLabeledExamples.future(to_iter(example_queue))
                example_queue.put(LabeledExampleRequest(searchId=self._context.search_id))

                if data_requirement is DataRequirement.DISTRIBUTED_FULL:
                    self._store_labeled_examples(examples,
                                                 lambda x: example_queue.put(LabeledExampleRequest(example=x)))
                else:
                    def add_example(example: LabeledExample) -> None:
                        if example.exampleSet.value is ExampleSet.TEST or example.label == '1':
                            example_queue.put(LabeledExampleRequest(example=example))

                    self._store_labeled_examples(examples, add_example)

                example_queue.put(None)
                future.result()
            else:
                self._store_labeled_examples(examples, None)

    @contextmanager
    def get_examples(self, example_set: ExampleSet) -> Iterable[Path]:
        with self._examples_lock:
            if self._context.node_index != 0:
                if example_set is ExampleSet.TRAIN:
                    assert self._get_data_requirement() is not DataRequirement.MASTER_ONLY

                self._sync_with_master(example_set)
                yield self._examples_dir / self._to_dir(example_set)
            else:
                example_dir = self._examples_dir / self._to_dir(example_set)
                if example_set is ExampleSet.TEST:
                    for label in example_dir.iterdir():
                        tmp_label_dir = self._tmp_dir / label.name
                        tmp_label_dir.mkdir(parents=True, exist_ok=True)
                        test_files = list(label.iterdir())
                        for i in range(0, len(test_files), len(self._context.nodes)):
                            test_files[i].rename(tmp_label_dir / test_files[i].name)

                    yield self._tmp_dir

                    for label in example_dir.iterdir():
                        for tmp_file in label.iterdir():
                            tmp_file.rename(example_dir / label.name / tmp_file.name)
                else:
                    yield example_dir

    def get_example_stream(self, example_set: ExampleSet, node_index: int) -> Iterable[ExampleMetadata]:
        assert self._examples_lock.locked()
        assert self._context.node_index == 0
        assert node_index != 0
        example_dir = self._examples_dir / self._to_dir(example_set)

        if example_set is ExampleSet.TRAIN:
            assert self._get_data_requirement() is not DataRequirement.MASTER_ONLY
            for label in example_dir.iterdir():
                if self._get_data_requirement() is DataRequirement.DISTRIBUTED_POSITIVES and label.name != '1':
                    continue

                for example in label.iterdir():
                    yield ExampleMetadata(label=label.name, key=example.name)
        elif example_set is ExampleSet.TEST:
            for label in example_dir.iterdir():
                label_examples = list(label.iterdir())
                for i in range(node_index - 1, len(label_examples), len(self._context.nodes) - 1):
                    yield ExampleMetadata(label=label.name, key=label_examples[i].name)
        else:
            raise NotImplementedError('Unknown example set: ' + self._to_dir(example_set))

    def get_example_path(self, example_set: ExampleSet, label: str, example: str) -> Path:
        assert self._examples_lock.locked()
        assert self._context.node_index == 0
        return self._examples_dir / self._to_dir(example_set) / label / example

    def reset(self, train_only: bool):
        with self._staging_lock:
            self._clear_dir(self._staging_dir, train_only)

        with self._examples_lock:
            self._clear_dir(self._examples_dir, train_only)

    def _clear_dir(self, dir_path: Path, train_only: bool):
        for child in dir_path.iterdir():
            if child.is_dir():
                if child.name != 'test' or not train_only:
                    self._clear_dir(child, train_only)
            else:
                child.unlink()

    def _store_labeled_examples(self, examples: Iterable[LabeledExample],
                                callback: Optional[Callable[[LabeledExample], None]]) -> None:
        with self._staging_lock:
            old_dirs = []
            for dir in self._staging_dir.iterdir():
                for label in dir.iterdir():
                    if label.name != IGNORE_FILE:
                        old_dirs.append(label)

            for example in examples:
                example_file = get_example_key(example.content)
                self._remove_old_paths(example_file, old_dirs)

                if example.label != '-1':
                    if example.HasField('exampleSet'):
                        example_subdir = self._to_dir(example.exampleSet.value)
                    else:
                        example_subdir = 'unspecified'

                    label_dir = self._staging_dir / example_subdir / example.label
                    label_dir.mkdir(parents=True, exist_ok=True)
                    example_path = label_dir / example_file
                    with example_path.open('wb') as f:
                        f.write(example.content)
                    logger.info('Saved example with label {} to path {}'.format(example.label, example_path))
                else:
                    logger.info('Example set to ignore - skipping')
                    ignore_file = self._staging_dir / IGNORE_FILE
                    with ignore_file.open('a+') as f:
                        f.write(example_file + '\n')

                if callback is not None:
                    callback(example)

        self._stored_examples_event.set()

    def _sync_with_master(self, example_set: ExampleSet) -> None:
        to_delete = defaultdict(set)
        example_dir = self._examples_dir / self._to_dir(example_set)
        for label in example_dir.iterdir():
            if self._get_data_requirement() is DataRequirement.DISTRIBUTED_POSITIVES \
                    and example_set is ExampleSet.TRAIN \
                    and label.name != '1':
                continue

            to_delete[label] = set(x.name for x in label.iterdir())

        for example in self._context.nodes[0].internal.GetExamples(
                GetExamplesRequest(searchId=self._context.search_id, exampleSet=example_set,
                                   nodeIndex=self._context.node_index)):
            if example.key in to_delete[example.label]:
                to_delete[example.label].remove(example.key)
            else:
                example_content = self._context.nodes[0].internal.GetExample(
                    GetExampleRequest(searchId=self._context.search_id, exampleSet=example_set, label=example.label,
                                      key=example.key))
                label_dir = example_dir / example.label
                label_dir.mkdir(parents=True, exist_ok=True)
                example_path = label_dir / example.key
                with example_path.open('wb') as f:
                    f.write(example_content.value)

        for label in to_delete:
            for file in to_delete[label]:
                example_path = label / file
                example_path.unlink()

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
                        example_dir = self._examples_dir / self._to_dir(example_set)
                        set_dirs[example_set] = list(example_dir.iterdir())

                    with self._staging_lock:
                        for file in self._staging_dir.iterdir():
                            if file.name == IGNORE_FILE:
                                with file.open() as ignore_file:
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

    def _promote_staging_examples_dir(self, subdir: Path, set_dirs: Dict['ExampleSet', List[Path]]) -> Tuple[int, int]:
        assert subdir.name == self._to_dir(ExampleSet.TRAIN) \
               or subdir.name == self._to_dir(ExampleSet.TEST) \
               or subdir.name == 'unspecified'

        new_positives = 0
        new_negatives = 0
        for label in subdir.iterdir():
            example_files = list(label.iterdir())
            if label.name == '1':
                new_positives += len(example_files)
            else:
                new_negatives += len(example_files)

            for example_file in example_files:
                for example_set in set_dirs:
                    if self._remove_old_paths(example_file.name, set_dirs[example_set]):
                        self._example_counts[example_set] -= 1

                if subdir.name == 'test' or (subdir.name == 'unspecified' and self._example_counts[ExampleSet.TEST] <
                                             self._example_counts[ExampleSet.TRAIN] * TEST_RATIO):
                    example_set = ExampleSet.TEST
                else:
                    example_set = ExampleSet.TRAIN

                self._example_counts[example_set] += 1
                example_dir = self._examples_dir / self._to_dir(example_set) / label.name
                example_dir.mkdir(parents=True, exist_ok=True)
                example_path = example_dir / example_file.name
                example_file.rename(example_path)
                logger.info('Promoted example with label {} to path {}'.format(label.name, example_path))

        return new_positives, new_negatives

    def _get_data_requirement(self) -> DataRequirement:
        return max([x.data_requirement for x in self._context.get_active_trainers()], key=lambda y: y.value)

    @staticmethod
    def _remove_old_paths(example_file: str, old_dirs: List[Path]) -> bool:
        for old_path in old_dirs:
            old_example_path = old_path / example_file
            if old_example_path.exists():
                old_example_path.unlink()
                logger.info('Removed old path {} for example'.format(old_example_path))
                return True

        return False

    @staticmethod
    def _to_dir(example_set: ExampleSet):
        return ExampleSet.Name(example_set).lower()
