import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class EfficientNetBaseline(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        weights = EfficientNet_B0_Weights.DEFAULT

        self.model = efficientnet_b0(weights=weights)

        # Replace original ImageNet classifier
        in_features = self.model.classifier[1].in_features

        self.model.classifier[1] = nn.Linear(
            in_features,
            num_classes
        )

    def forward(self, x):
        return self.model(x)