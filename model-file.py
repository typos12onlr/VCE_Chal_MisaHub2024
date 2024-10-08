# %% [code]
import torch
from torchvision import models

model = models.resnet50(weights='IMAGENET1K_V1')
