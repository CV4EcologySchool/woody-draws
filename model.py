import torch.nn as nn
from torchvision.models import resnet
from torchvision.models.resnet import BasicBlock, Bottleneck

class NAIPResNet50(resnet.ResNet):

    def __init__(self,block, layers,n_classes, n_channels):
        '''
            Constructor of the model. Here, we initialize the model's
            architecture (layers).
        '''
        super(NAIPResNet50, self).__init__(block, layers, n_classes)
        #self.n_classes = num_classes
        self.conv1 = nn.Conv2d(
            n_channels,
            out_channels=64,
            kernel_size=7,
            stride=1,
            padding=2,
            bias=False,
        )

        self.feature_extractor = resnet.resnet50(pretrained=True)       # "pretrained": use weights pre-trained on ImageNet
        # replace the very last layer from the original, 1000-class output
        # ImageNet to a new one that outputs num_classes
        #last_layer = self.feature_extractor.fc                          # tip: print(self.feature_extractor) to get info on how model is set up
        #in_features = last_layer.in_features                            # number of input dimensions to last (classifier) layer
        #self.feature_extractor.fc = nn.Identity()                       # discard last layer...

        #self.classifier = nn.Linear(in_features, n_classes)           # ...and create a new one

#    def forward(self, x):
#        '''
#            Forward pass. Here, we define how to apply our model. It's basically
#            applying our modified ResNet-18 on the input tensor ("x") and then
#            apply the final classifier layer on the ResNet-18 output to get our
#            num_classes prediction.
#        '''
#        # x.size(): [B x 3 x W x H]
#        features = self.feature_extractor(x)    # features.size(): [B x 512 x W x H]
#        prediction = self.classifier(features)  # prediction.size(): [B x num_classes]

#        return prediction
