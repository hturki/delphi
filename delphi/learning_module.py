import logging
import signal
import sys
import threading
import traceback
from concurrent import futures
from pathlib import Path

import grpc
import logzero
import multiprocessing_logging
import yaml
from logzero import logger

from delphi.condition.examples_per_label_condition import ExamplesPerLabelCondition
from delphi.condition.test_auc_condition import TestAucCondition
from delphi.coordinator import Coordinator
from delphi.learning_module_stub import LearningModuleStub
from delphi.mpncov.mpncov_trainer import MPNCovTrainer
from delphi.proto import learning_module_pb2_grpc, internal_pb2_grpc, diamond_pb2_grpc, admin_pb2_grpc
from delphi.retrain.absolute_threshold_policy import AbsoluteThresholdPolicy
from delphi.retrain.percentage_threshold_policy import PercentageThresholdPolicy
from delphi.servicer.admin_servicer import AdminServicer
from delphi.servicer.diamond_servicer import DiamondServicer
from delphi.servicer.internal_servicer import InternalServicer
from delphi.servicer.learning_module_servicer import LearningModuleServicer
from delphi.svm.distributed_svm_trainer import DistributedSVMTrainer
from delphi.svm.ensemble_svm_trainer import EnsembleSVMTrainer
from delphi.svm.fs_feature_cache import FSFeatureCache
from delphi.svm.redis_feature_cache import RedisFeatureCache
from delphi.svm.svm_trainer import SVMTrainer
from delphi.utils import log_exceptions
from delphi.wsdan.wsdan_trainer import WSDANTrainer

logzero.loglevel(logging.INFO)


def dumpstacks(_, __):
    traceback.print_stack()
    id2name = dict([(th.ident, th.name) for th in threading.enumerate()])
    code = []
    for threadId, stack in sys._current_frames().items():
        code.append("\n# Thread: %s(%d)" % (id2name.get(threadId, ""), threadId))
        for filename, lineno, name, line in traceback.extract_stack(stack):
            code.append('File: "%s", line %d, in %s' % (filename, lineno, name))
            if line:
                code.append("  %s" % (line.strip()))
    print("\n".join(code))


@log_exceptions
def main():
    multiprocessing_logging.install_mp_handler()

    config_path = sys.argv[1] if len(sys.argv) > 1 else (Path.home() / '.delphi' / 'config.yml')

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if config.get('debug', False):
        signal.signal(signal.SIGUSR1, dumpstacks)

    node_index = config['node_index']
    nodes = config['nodes']
    port = config['port']
    retrain_policy_config = config['retrain_policy']
    only_positives = retrain_policy_config['only_positives']
    threshold = retrain_policy_config['threshold']
    if retrain_policy_config['type'] == 'absolute':
        retrain_policy = AbsoluteThresholdPolicy(threshold, only_positives)
    elif retrain_policy_config['type'] == 'percentage':
        retrain_policy = PercentageThresholdPolicy(threshold, only_positives)
    else:
        raise NotImplementedError('unknown retrain policy: {}'.format(retrain_policy_config))

    only_use_better_models = config['only_use_better_models']
    root_dir = config['root_dir']

    coordinator = Coordinator(node_index, [LearningModuleStub(node, port) for node in nodes], retrain_policy,
                              only_use_better_models, root_dir)

    model_dir = config['model_dir']

    trainers = []

    trainer_configs = config['train_strategy']
    for i in range(len(trainer_configs)):
        model = trainer_configs[i]['model']
        if model['type'] == 'svm':
            feature_extractor = model['feature_extractor']
            probability = model['probability']
            linear_only = model['linear_only']

            cache_config = model['cache']
            if cache_config['type'] == 'redis':
                feature_cache = RedisFeatureCache(cache_config['port'])
            elif cache_config['type'] == 'filesystem':
                feature_cache = FSFeatureCache(cache_config['feature_dir'])
            else:
                raise NotImplementedError('unknown feature cache type: {}'.format(cache_config))

            mode = model['mode']
            if mode == 'master_only':
                trainer = SVMTrainer(coordinator, model_dir, feature_extractor, feature_cache, probability,
                                     linear_only)
            elif mode == 'distributed':
                trainer = DistributedSVMTrainer(coordinator, model_dir, feature_extractor, feature_cache,
                                                probability, linear_only, i)
            elif mode == 'ensemble':
                assert probability
                trainer = EnsembleSVMTrainer(coordinator, model_dir, feature_extractor, feature_cache,
                                             linear_only)
            else:
                raise NotImplementedError('unknown svm mode: {}'.format(mode))
        elif model['type'] == 'fast_mpncov':
            trainer = MPNCovTrainer(coordinator, model['distributed'], model['freeze'], model_dir)
        elif model['type'] == 'wsdan':
            trainer = WSDANTrainer(coordinator, model['distributed'], model['visualize'], model['freeze'])
        else:
            raise NotImplementedError('unknown model: {}'.format(model))

        condition = trainer_configs[i]['condition']
        if condition['type'] == 'examples_per_label':
            trainers.append(ExamplesPerLabelCondition(condition['count'], trainer))
        elif condition['type'] == 'test_auc':
            trainers.append(TestAucCondition(condition['threshold'], trainer))
        else:
            raise NotImplementedError('unknown condition: {}'.format(condition))

    coordinator.trainers = trainers
    logger.info('Starting learning module as node {}: {}'.format(node_index, __file__))
    server = grpc.server(futures.ThreadPoolExecutor(), options=[
        ('grpc.max_send_message_length', 1024 * 1024 * 1024),
        ('grpc.max_receive_message_length', 1024 * 1024 * 1024)
    ])
    learning_module_pb2_grpc.add_LearningModuleServiceServicer_to_server(LearningModuleServicer(coordinator),
                                                                         server)
    internal_pb2_grpc.add_InternalServiceServicer_to_server(InternalServicer(coordinator), server)
    diamond_pb2_grpc.add_DiamondServiceServicer_to_server(DiamondServicer(coordinator), server)
    admin_pb2_grpc.add_AdminServiceServicer_to_server(AdminServicer(coordinator), server)

    server.add_insecure_port('0.0.0.0:{}'.format(port))
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    main()
