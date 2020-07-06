import torch.nn as nn

__all__ = ['Newmodel', 'get_model']

from delphi.mpncov.base import Basemodel
from delphi.mpncov.mpncov import MPNCOV


class Newmodel(Basemodel):
    def __init__(self, num_classes, freezed_layer, model_dir, extract_feature_vector):
        super(Newmodel, self).__init__(model_dir, extract_feature_vector)
        self.representation = MPNCOV(input_dim=256)
        fc_input_dim = self.representation.output_dim

        self.classifier = nn.Linear(fc_input_dim, num_classes)
        index_before_freezed_layer = 0
        if freezed_layer:
            for m in self.features.children():
                if index_before_freezed_layer < freezed_layer:
                    self._freeze(m)
                index_before_freezed_layer += 1

    def _freeze(self, modules):
        for param in modules.parameters():
            param.requires_grad = False
        return modules


def get_model(num_classes, freezed_layer, model_dir, extract_feature_vector):
    _model = Newmodel(num_classes, freezed_layer, model_dir, extract_feature_vector)
    return _model
