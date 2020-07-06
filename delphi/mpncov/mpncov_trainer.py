import copy
import io
import time
from typing import Optional

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from logzero import logger
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import WeightedRandomSampler
from torchvision import datasets
from tqdm import tqdm

from delphi.context.model_trainer_context import ModelTrainerContext
from delphi.model import Model
from delphi.mpncov.model_init import get_model
from delphi.mpncov.mpncov_model import MPNCovModel, MPNCOV_TEST_TRANSFORMS
from delphi.pytorch.distributed_proxy_sampler import DistributedProxySampler
from delphi.pytorch.pytorch_trainer_base import PytorchTrainerBase
from delphi.utils import get_weights, AverageMeter

TRAIN_BATCH_SIZE = 24


class MPNCovTrainer(PytorchTrainerBase):

    def __init__(self, context: ModelTrainerContext, distributed: bool, freeze: Optional[int], model_dir: str):
        super().__init__(context, MPNCOV_TEST_TRANSFORMS, distributed)

        self._curr_epoch = 0
        self._global_step = 0

        self._model = get_model(2, 0, model_dir, False).to(self.device, non_blocking=True)
        if freeze is not None:
            children = [c for c in self._model.children()]
            freeze = len(children) - freeze
            for depth, child in enumerate(children):
                if depth < freeze:
                    for param in child.parameters():
                        param.requires_grad = False

        if distributed:
            self._model = DistributedDataParallel(self._model)

        self._model.train()

        # setup optimizer
        lr = 0.00794328234
        weight_decay = 0.0001
        params_list = [{'params': self._model.features.parameters(),
                        'lr': lr,
                        'weight_decay': weight_decay},
                       {'params': self._model.representation.parameters(),
                        'lr': lr,
                        'weight_decay': weight_decay},
                       {'params': self._model.classifier.parameters(),
                        'lr': lr * 5,
                        'weight_decay': weight_decay}]

        self._optimizer = torch.optim.SGD(params_list, lr=lr, momentum=0.9, weight_decay=weight_decay)
        self._criterion = nn.CrossEntropyLoss().to(self.device, non_blocking=True)

        self._train_transforms = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(448),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def load_from_file(self, model_version: int, file: bytes) -> Model:
        bytes = io.BytesIO()
        bytes.write(file)
        bytes.seek(0)
        checkpoint = torch.load(bytes)
        self._model.load_state_dict(checkpoint['state_dict'])
        self._curr_epoch = checkpoint['epoch']
        self._optimizer.load_state_dict(checkpoint['optimizer'])

        model_pred = copy.deepcopy(self._model)
        model_pred.eval()
        return MPNCovModel(model_pred, self.get_new_version(), self._curr_epoch, self._optimizer.state_dict(),
                           self.pool)

    def train_model(self, train_dir: str) -> Model:
        start_time = time.time()
        start_epoch = self._curr_epoch
        epochs = 100
        end_epoch = start_epoch + epochs
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        dataset = datasets.ImageFolder(train_dir, transform=self._train_transforms)
        weights = get_weights(dataset.targets)
        sampler = WeightedRandomSampler(weights, len(weights))
        if self.distributed:
            sampler = DistributedProxySampler(sampler)

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, sampler=sampler, num_workers=8)

        train_image_count = len(train_loader)
        for epoch in tqdm(range(start_epoch, end_epoch)):
            self._curr_epoch += 1

            end = time.time()
            for i, (input, target) in enumerate(train_loader):
                data_time.update(time.time() - end)
                # measure data loading time
                input = input.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                # compute output
                output = self._model(input)
                loss = self._criterion(output, target)

                # measure accuracy and record loss
                losses.update(loss.item(), input.size(0))

                # compute gradient and do SGD step
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % 10 == 0:
                    self.context.tb_writer.add_scalar('train/loss', loss.item(),
                                                      self._curr_epoch * train_image_count + i)

                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f}))'
                          .format(epoch, i, train_image_count, batch_time=batch_time, data_time=data_time, loss=losses))

        end_time = time.time()
        logger.info('Trained model for {} epochs in {:.3f} seconds'.format(epochs, end_time - start_time))

        model_pred = copy.deepcopy(self._model)
        model_pred.eval()

        return MPNCovModel(model_pred, self.get_new_version(), self._curr_epoch, self._optimizer.state_dict(),
                           self.pool)
