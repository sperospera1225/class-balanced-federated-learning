import timm
import torch
import torch.nn as nn

class Resnet50(nn.Module):
    def __init__(self,):
        super().__init__()
        num_class = 10
        self.backbone = timm.create_model("resnet50",pretrained=True)
        self.backbone.conv1 = nn.Conv2d(1,64,kernel_size=7,stride=2,padding=2,bias=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features,num_class,bias=True)
    def forward(self,img):
        result = self.backbone(img)
        return result
if __name__ == '__main__':
    num_class = 10
    x = torch.randn(100,1,28,28)
    backbone = timm.create_model("resnet50",pretrained=True)
    backbone.conv1 = nn.Conv2d(1,64,kernel_size=7,stride=2,padding=2,bias=False)
    backbone.fc = nn.Linear(backbone.fc.in_features,num_class,bias=True)
    features = backbone(x)