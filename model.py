import torch.nn as nn
from torchvision.models import resnet
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision import models



def _resnet_backbone_maker(backbone: str, **kwargs) -> resnet.ResNet:
    return models.__dict__[backbone](**kwargs)

def define_resnet(cfg: dict, backbone: str, **kwargs) -> resnet.ResNet:
    model = _resnet_backbone_maker(backbone, **kwargs)
    model.conv1 = nn.Conv2d(cfg["n_channels"],out_channels=64,kernel_size=7,stride=1,padding=2,bias=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, cfg["num_classes"])
    return model








class NAIPResNet50(resnet.ResNet):

    def __init__(self, block, layers, n_classes, model):
        '''
            Constructor of the model. Here, we initialize the model's
            architecture (layers).
        '''
        #super(NAIPResNet50, self).__init__(n_classes)
        super(NAIPResNet50, self).__init__(block, layers, n_classes)
        #self.n_classes = num_classes
        #self.model = models.__dict__["resnet50"](strict=False)
        #self.model = models.resnet50(pretrained=False)
        #self.model = model
        self.model.conv1 = nn.Conv2d(
            4,
            out_channels=64,
            kernel_size=7,
            stride=1,
            padding=2,
            bias=False,
        )
        #weight = model.conv1.weight.clone()
        #self.conv1.weight[:,:3,:,:] = nn.Parameter(weight)
        #self.conv1.weight[:,3,:,:] = nn.Parameter(weight[:,1,:,:]) 
        num_ftrs = self.model.fc.in_features
        model.fc = nn.Linear(num_ftrs, n_classes)

