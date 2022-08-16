import torch.nn as nn
from torchvision.models import resnet
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision import models



def _resnet_backbone_maker(backbone: str, **kwargs) -> resnet.ResNet:
    """
    Return the standard resnet model with backbone and nlayers specified
    """
    return models.__dict__[backbone](**kwargs)

def define_resnet(cfg: dict, backbone: str, **kwargs) -> resnet.ResNet:
    """
    instantiate model with backbone and number of channels and output classes
    as specified by config file.
    """
    model = _resnet_backbone_maker(backbone, **kwargs)
    
    model.conv1 = nn.Conv2d(cfg["n_channels"],out_channels=64,kernel_size=7,stride=1,padding=2,bias=False)
    
    num_ftrs = model.fc.in_features
    
    model.fc = nn.Linear(num_ftrs, cfg["num_classes"])
    
    return model
