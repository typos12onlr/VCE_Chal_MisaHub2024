# %% [code] {"execution":{"iopub.status.busy":"2024-10-21T05:11:05.571989Z","iopub.execute_input":"2024-10-21T05:11:05.572404Z","iopub.status.idle":"2024-10-21T05:11:05.577310Z","shell.execute_reply.started":"2024-10-21T05:11:05.572364Z","shell.execute_reply":"2024-10-21T05:11:05.576381Z"}}
# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"execution":{"iopub.status.busy":"2024-10-21T05:21:56.334175Z","iopub.execute_input":"2024-10-21T05:21:56.334720Z","iopub.status.idle":"2024-10-21T05:22:09.636412Z","shell.execute_reply.started":"2024-10-21T05:21:56.334682Z","shell.execute_reply":"2024-10-21T05:22:09.634805Z"}}
# !pip install vit-pytorch

# %% [code] {"execution":{"iopub.status.busy":"2024-10-21T05:23:34.097135Z","iopub.execute_input":"2024-10-21T05:23:34.098138Z","iopub.status.idle":"2024-10-21T05:23:34.112321Z","shell.execute_reply.started":"2024-10-21T05:23:34.098089Z","shell.execute_reply":"2024-10-21T05:23:34.111231Z"}}
import torch
import torch.nn as nn
import torchvision.models as models
import torchinfo
from vit_pytorch import ViT

# %% [code] {"execution":{"iopub.status.busy":"2024-10-21T05:30:04.966006Z","iopub.execute_input":"2024-10-21T05:30:04.966450Z","iopub.status.idle":"2024-10-21T05:30:04.973280Z","shell.execute_reply.started":"2024-10-21T05:30:04.966406Z","shell.execute_reply":"2024-10-21T05:30:04.971966Z"}}
class CNNBranch(nn.Module):
    def __init__(self, out_dim=512):
        super(CNNBranch, self).__init__()
        #Using a pre-trained ResNet as the CNN backbone (can replace with any other CNN)
        self.cnn = models.resnet18(pretrained=True)
        # Modifying the final layer to output a specific size of features
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, out_dim)  # Output size: 512 features

    def forward(self, x):
        # Output shape: (batch_size, out_dim)
        return self.cnn(x)

# %% [code] {"execution":{"iopub.status.busy":"2024-10-21T05:31:05.044603Z","iopub.execute_input":"2024-10-21T05:31:05.045527Z","iopub.status.idle":"2024-10-21T05:31:05.052448Z","shell.execute_reply.started":"2024-10-21T05:31:05.045475Z","shell.execute_reply":"2024-10-21T05:31:05.051130Z"}}
class ViTBranch(nn.Module):
    def __init__(self, image_size=224, patch_size=16, out_dim=512):
        super(ViTBranch, self).__init__()
        # Vision Transformer
        self.vit = ViT(
            image_size=image_size,  # Input image size
            patch_size=patch_size,  # Patch size
            num_classes=out_dim,  # Output 512 features, same as CNN branch
            dim=512,  # Embedding dimension
            depth=6,  # Number of transformer blocks
            heads=8,  # Attention heads
            mlp_dim=1024  # MLP (Feed-Forward Network) size
        )

    def forward(self, x):
        # Output shape: (batch_size, out_dim)
        return self.vit(x)

# %% [code] {"execution":{"iopub.status.busy":"2024-10-21T05:31:05.728083Z","iopub.execute_input":"2024-10-21T05:31:05.729031Z","iopub.status.idle":"2024-10-21T05:31:05.736884Z","shell.execute_reply.started":"2024-10-21T05:31:05.728974Z","shell.execute_reply":"2024-10-21T05:31:05.735649Z"}}
class ClassificationModel(nn.Module):
    def __init__(self, num_classes=10, feature_dim=512):
        super(ClassificationModel, self).__init__()
        self.cnn_branch = CNNBranch(out_dim= feature_dim)  # CNN branch
        self.vit_branch = ViTBranch(out_dim= feature_dim)  # ViT branch
        
        # Combine both branches' features
        self.combined_fc = nn.Sequential(
            nn.Linear(1024, 512),  # Combine 512 + 512 from CNN and ViT
            nn.ReLU(),
            nn.Linear(512, 256),  # Reduce features to 256
            nn.ReLU(),
            nn.Linear(256, num_classes)  # Output number of classes
        )

    def forward(self, x):
        # Extract features from both branches
        cnn_features = self.cnn_branch(x)  # Shape: (batch_size, feature_dim)
        vit_features = self.vit_branch(x)  # Shape: (batch_size, feature_dim)
        
        # Concatenate the features from both branches
        combined_features = torch.cat((cnn_features, vit_features), dim=1)  # Shape: (batch_size, 1024)
        
        # Pass through the fully connected layers
        output = self.combined_fc(combined_features)  # Shape: (batch_size, num_classes)
        return output


# %% [code] {"execution":{"iopub.status.busy":"2024-10-21T05:31:07.140339Z","iopub.execute_input":"2024-10-21T05:31:07.141280Z","iopub.status.idle":"2024-10-21T05:31:08.697081Z","shell.execute_reply.started":"2024-10-21T05:31:07.141208Z","shell.execute_reply":"2024-10-21T05:31:08.695776Z"}}
# model = CombinedModel(num_classes=10)  # Assuming 10 classes
# input_image = torch.randn(8, 3, 224, 224)  # Example batch of images (batch_size=8, 3 channels, 224x224)
# output = model(input_image)
# print(output.shape)  # Output: (batch_size, 10)

## prints [8,10]

# %% [code] {"execution":{"iopub.status.busy":"2024-10-21T05:31:16.065186Z","iopub.execute_input":"2024-10-21T05:31:16.065620Z","iopub.status.idle":"2024-10-21T05:31:16.070539Z","shell.execute_reply.started":"2024-10-21T05:31:16.065577Z","shell.execute_reply":"2024-10-21T05:31:16.069325Z"}}
# summary = torchinfo.summary(model, (8,3,224,224))
# print(summary)

# %% [code] {"execution":{"iopub.status.busy":"2024-10-21T05:26:36.871509Z","iopub.execute_input":"2024-10-21T05:26:36.871930Z","iopub.status.idle":"2024-10-21T05:26:36.877010Z","shell.execute_reply.started":"2024-10-21T05:26:36.871892Z","shell.execute_reply":"2024-10-21T05:26:36.875774Z"}}


# %% [code]
