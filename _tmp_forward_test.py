import sys
import torch
sys.path.insert(0, 'src')
from models.backbones.resnet import ResNet18_BoT

model = ResNet18_BoT(num_classes=4, heads=4, pretrained=False, dropout=0.0)
model.eval()

x = torch.randn(2,3,224,224)
with torch.no_grad():
    out = model(x)
print('output ok, shape=', out.shape)
