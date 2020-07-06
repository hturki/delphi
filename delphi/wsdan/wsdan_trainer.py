import copy
import io
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from logzero import logger
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from tqdm import tqdm

from delphi.context.model_trainer_context import ModelTrainerContext
from delphi.model import Model
from delphi.pytorch.distributed_proxy_sampler import DistributedProxySampler
from delphi.pytorch.pytorch_trainer_base import PytorchTrainerBase
from delphi.utils import AverageMeter, get_weights
from delphi.wsdan.wsdan import WSDAN
from delphi.wsdan.wsdan_common import batch_augment
from delphi.wsdan.wsdan_model import WSDANModel, WSDAN_TEST_TRANSFORMS

TRAIN_BATCH_SIZE = 16


class WSDANTrainer(PytorchTrainerBase):

    def __init__(self, context: ModelTrainerContext, distributed: bool, visualize: bool, freeze: Optional[int]):
        super().__init__(context, WSDAN_TEST_TRANSFORMS, distributed)
        self._visualize = visualize

        self._curr_epoch = 0
        self._global_step = 0

        self._model = WSDAN(num_classes=2, pretrained=True).to(self.device, non_blocking=True)

        if freeze is not None:
            children = [c for c in self._model.children()]
            freeze = len(children) - freeze
            for depth, child in enumerate(children):
                if depth < freeze:
                    for param in child.parameters():
                        param.requires_grad = False

        # feature_center: size of (#classes, #attention_maps * #channel_features)
        self._feature_center = torch.zeros(2, 32 * self._model.num_features).to(self.device, non_blocking=True)

        if distributed:
            self._model = DistributedDataParallel(nn.SyncBatchNorm.convert_sync_batchnorm(self._model))

        self._model.train()

        # setup optimizer
        self._optimizer = torch.optim.SGD(self._model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-5)
        self._scheduler = torch.optim.lr_scheduler.StepLR(self._optimizer, step_size=2, gamma=0.9)
        self._cross_entropy_loss = nn.CrossEntropyLoss().to(self.device, non_blocking=True)
        self._center_loss = CenterLoss().to(self.device)

        self._train_transforms = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(448),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.126, saturation=0.5),
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
        self._feature_center = checkpoint['feature_center'].to(self.device, non_blocking=True)

        model_pred = copy.deepcopy(self._model)
        model_pred.eval()

        return WSDANModel(model_pred, model_version, self._get_image_writer(), self._curr_epoch, checkpoint['optimizer'],
                          checkpoint['feature_center'], self.pool)

    def train_model(self, train_dir: str) -> Model:
        start_time = time.time()
        start_epoch = self._curr_epoch
        epochs = 10

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
                self._optimizer.zero_grad()

                data_time.update(time.time() - end)

                # measure data loading time
                input = input.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                ##################################
                # Raw Image
                ##################################
                # compute output
                y_pred_raw, feature_matrix, attention_map = self._model(input)

                feature_center_batch = F.normalize(self._feature_center[target], dim=-1)
                self._feature_center[target] += 5e-2 * (feature_matrix.detach() - feature_center_batch)

                ##################################
                # Attention Cropping
                ##################################
                with torch.no_grad():
                    crop_images = batch_augment(input, attention_map[:, :1, :, :], mode='crop', theta=(0.4, 0.6),
                                                padding_ratio=0.1)

                # crop images forward
                y_pred_crop, _, _ = self._model(crop_images)

                ##################################
                # Attention Dropping
                ##################################
                with torch.no_grad():
                    drop_images = batch_augment(input, attention_map[:, 1:, :, :], mode='drop', theta=(0.2, 0.5))

                # drop images forward
                y_pred_drop, _, _ = self._model(drop_images)

                # loss
                batch_loss = self._cross_entropy_loss(y_pred_raw, target) / 3. + \
                             self._cross_entropy_loss(y_pred_crop, target) / 3. + \
                             self._cross_entropy_loss(y_pred_drop, target) / 3. + \
                             self._center_loss(feature_matrix, feature_center_batch)

                # backward
                batch_loss.backward()
                self._optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                losses.update(batch_loss.item(), input.size(0))

                if i % 10 == 0:
                    self.context.tb_writer.add_scalar('train/loss', batch_loss.item(),
                                                      self._curr_epoch * train_image_count + i)

                    logger.info('Epoch: [{0}][{1}/{2}]\t'
                                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                                'Loss {loss.val:.4f} ({loss.avg:.4f}))'
                                .format(epoch, i, train_image_count, batch_time=batch_time, data_time=data_time,
                                        loss=losses))

            self._scheduler.step()

        end_time = time.time()
        logger.info(
            'Trained model for {} epochs in {:.3f} seconds'.format(self._curr_epoch - start_epoch,
                                                                   end_time - start_time))

        model_pred = copy.deepcopy(self._model)
        model_pred.eval()

        return WSDANModel(model_pred, self.get_new_version(), self._get_image_writer(), self._curr_epoch,
                          self._optimizer.state_dict(), self._feature_center.cpu(), self.pool)

    def _get_image_writer(self) -> Optional[SummaryWriter]:
        return self.context.tb_writer if self._visualize else None


class CenterLoss(nn.Module):
    def __init__(self):
        super(CenterLoss, self).__init__()
        self.l2_loss = nn.MSELoss(reduction='sum')

    def forward(self, outputs, targets):
        return self.l2_loss(outputs, targets) / outputs.size(0)
