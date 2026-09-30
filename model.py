import torchvision
from torch import nn


def create_effnetb2_model(num_classes=3):
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    transforms = weights.transforms()  # no download needed for transforms

    # weights=None: skip the ImageNet download, our .pth has all the weights
    model = torchvision.models.efficientnet_b2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1408, out_features=num_classes),
    )
    return model, transforms
