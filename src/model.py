import torch.nn as nn
from torchvision import models

def get_model(num_classes=2, pretrained = True, freeze=True):

    # Load DenseNet
    model = models.densenet121(pretrained=pretrained)

    # Freeze backbone
    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    # replace classifier
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.classifier.in_features, num_classes)
    )

    # Ensure classifier is trainable
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


