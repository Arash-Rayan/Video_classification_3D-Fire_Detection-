from torch import nn 

from torchvision.models.video import r3d_18 , R3D_18_Weights
resnet_model = r3d_18(weights=R3D_18_Weights.DEFAULT)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 1)  # Binary classification

resnet_model.fc