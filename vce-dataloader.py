# %% [code]
# %% [code] {"jupyter":{"outputs_hidden":false}}

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-08T06:19:36.274243Z","iopub.execute_input":"2024-10-08T06:19:36.274704Z","iopub.status.idle":"2024-10-08T06:19:36.281537Z","shell.execute_reply.started":"2024-10-08T06:19:36.274661Z","shell.execute_reply":"2024-10-08T06:19:36.280111Z"}}
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torchvision.utils import make_grid

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-08T06:19:36.508239Z","iopub.execute_input":"2024-10-08T06:19:36.508646Z","iopub.status.idle":"2024-10-08T06:19:36.515892Z","shell.execute_reply.started":"2024-10-08T06:19:36.508608Z","shell.execute_reply":"2024-10-08T06:19:36.514490Z"}}
class BinaryDataset(Dataset):
    def __init__(self, dataset, target_class_idx):
        self.dataset = dataset
        self.target_class_idx = target_class_idx
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        binary_label = 1 if label == self.target_class_idx else 0
        return image, binary_label

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-08T06:31:40.383731Z","iopub.execute_input":"2024-10-08T06:31:40.384214Z","iopub.status.idle":"2024-10-08T06:31:40.395654Z","shell.execute_reply.started":"2024-10-08T06:31:40.384168Z","shell.execute_reply":"2024-10-08T06:31:40.394214Z"}}
def getBinaryDataLoader(image_size = (224,224), target_class_name = None, path_to_dataset = None, batch_size = 32, sampling = True):
    
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])
    
    dataset = datasets.ImageFolder(root = path_to_dataset, transform = transform)
    
    target_class = target_class_name
    
    target_class_idx = dataset.class_to_idx[target_class]
    
    binary_dataset = BinaryDataset(dataset, target_class_idx)
    targets = np.array([label for _, label in binary_dataset])
    class_sample_counts = np.array([(targets == 0).sum(), (targets == 1).sum()])
    class_sample_counts = torch.tensor(class_sample_counts)
    class_weights = 1. / class_sample_counts.float()
    sample_weights = [class_weights[label] for label in targets]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    if sampling:
        binaryDataLoader = DataLoader(binary_dataset, batch_size=batch_size, shuffle = False, sampler = sampler)
    else:
        binaryDataLoader = DataLoader(binary_dataset, batch_size=batch_size, shuffle = True)
    return binaryDataLoader

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-08T06:31:45.502099Z","iopub.execute_input":"2024-10-08T06:31:45.502701Z","iopub.status.idle":"2024-10-08T06:31:45.513883Z","shell.execute_reply.started":"2024-10-08T06:31:45.502643Z","shell.execute_reply":"2024-10-08T06:31:45.512413Z"}}
def getAllDataLoader(image_size = (224,224), path_to_dataset = None, batch_size = 32, sampling = True):
    transform = transforms.Compose([
        transforms.Resize(image_size),  # Adjust size based on your model
        transforms.ToTensor(),
    ])
    all_classes_dataset = datasets.ImageFolder(root=path_to_dataset, transform=transform)
    targets = np.array([label for _, label in all_classes_dataset])
    class_sample_counts = torch.tensor(np.array([(targets == i).sum() for i in range(len(all_classes_dataset.classes))]))
    class_weights = 1. / class_sample_counts.float()
    sample_weights = [class_weights[label] for label in targets]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    if sampling:
        all_classes_dataloader = DataLoader(all_classes_dataset, batch_size=batch_size, shuffle=False, sampler = sampler)
    else:
        all_classes_dataloader = DataLoader(all_classes_dataset, batch_size=batch_size, shuffle=True)
    return all_classes_dataloader

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-08T06:19:38.350439Z","iopub.execute_input":"2024-10-08T06:19:38.350980Z","iopub.status.idle":"2024-10-08T06:19:38.360735Z","shell.execute_reply.started":"2024-10-08T06:19:38.350906Z","shell.execute_reply":"2024-10-08T06:19:38.359301Z"}}
def visualize_batch(dataloader, nrow = 8):
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    plt.figure(figsize=(10,10))
    imshow(make_grid(images, nrow = nrow))
#     print('Labels:', labels)
#     # If the dataset has a class_to_idx attribute, map labels back to class names
#     if hasattr(dataloader.dataset, 'classes'):
#         class_names = [dataloader.dataset.classes[label] for label in labels]
#         print('Class names:', class_names)

# Helper function to unnormalize and display the image grid
def imshow(img):
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))  # Convert from CHW to HWC
    plt.show()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-08T06:31:49.600505Z","iopub.execute_input":"2024-10-08T06:31:49.601045Z","iopub.status.idle":"2024-10-08T06:34:27.782252Z","shell.execute_reply.started":"2024-10-08T06:31:49.600993Z","shell.execute_reply":"2024-10-08T06:34:27.781094Z"}}
if __name__ == '__main__':
    binDL = getBinaryDataLoader(image_size=(224,224),target_class_name="Normal", 
                                path_to_dataset="/kaggle/input/vce-dataset/training", batch_size= 32)
    visualize_batch(binDL)
    
    allDL = getAllDataLoader(image_size=(224,224), path_to_dataset="/kaggle/input/vce-dataset/training", batch_size=32)
    visualize_batch(allDL)