import torch.nn as nn
import torchvision.models as models

class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        # Load MobileNetV2 pretrained on ImageNet
        mobilenet_v2 = models.mobilenet_v2(pretrained=True)
        
        # Remove the fully connected layers of MobileNetV2
        self.features = mobilenet_v2.features
        
        # Add your own classifier layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)  # 1280 is the output channels of MobileNetV2's last layer
        )

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])  # Global average pooling
        x = self.classifier(x)
        return x