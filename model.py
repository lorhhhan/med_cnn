from efficientnet_pytorch import EfficientNet
import torch.nn as nn

def build_model(num_classes):
    model = EfficientNet.from_pretrained('efficientnet-b4')
    model._fc = nn.Linear(model._fc.in_features, num_classes)
    return model
