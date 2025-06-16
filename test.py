from torchinfo import summary 
import torch 
from torch import nn 

slow_fast_model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50' , pretrained=True)

# summary(slow_fast_model)
in_features = slow_fast_model.blocks[-1].proj.in_features
# nn.linear(in_feature , 1)
slow_fast_model.blocks[-1].proj = nn.Linear(in_features , 1)

print(slow_fast_model.blocks)