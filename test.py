# import torch
# from torchvision.transforms import Compose
# from pytorchvideo.transforms import (
#     ApplyTransformToKey,
#     UniformTemporalSubsample,
#     ShortSideScale,
#     UniformCropVideo
# )

# transform = ApplyTransformToKey(
#     key="video",
#     transform=Compose([
#         UniformTemporalSubsample(8),     
#         ShortSideScale(256),             
#         UniformCropVideo(224, 0),        
#     ])
# )

import albumentations as A
 