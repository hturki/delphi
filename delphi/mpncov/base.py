import torch.nn as nn

from delphi.mpncov.mpncovresnet import mpncovresnet50


class Basemodel(nn.Module):
    """Load backbone model and reconstruct it into three part:
       1) feature extractor
       2) global image representaion
       3) classifier
    """

    def __init__(self, model_dir, extract_feature_vector):
        super(Basemodel, self).__init__()
        basemodel = mpncovresnet50(model_dir)
        basemodel = self._reconstruct_mpncovresnet(basemodel)
        self.features = basemodel.features
        self.representation = basemodel.representation
        self.classifier = basemodel.classifier
        self.representation_dim = basemodel.representation_dim
        self.extract_feature_vector = extract_feature_vector

    def forward(self, x):
        x = self.features(x)
        x = self.representation(x)
        x = x.view(x.size(0), -1)

        if self.extract_feature_vector:
            return x
        else:
            return self.classifier(x)

    @staticmethod
    def _reconstruct_mpncovresnet(basemodel):
        model = nn.Module()
        model.features = nn.Sequential(*list(basemodel.children())[:-1])
        model.representation_dim = basemodel.layer_reduce.weight.size(0)
        model.representation = None
        model.classifier = basemodel.fc
        return model