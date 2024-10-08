# %% [code]
# %% [code] {"execution":{"iopub.status.busy":"2024-10-07T18:28:41.161242Z","iopub.execute_input":"2024-10-07T18:28:41.161932Z","iopub.status.idle":"2024-10-07T18:28:46.507300Z","shell.execute_reply.started":"2024-10-07T18:28:41.161886Z","shell.execute_reply":"2024-10-07T18:28:46.505764Z"}}
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torchvision.utils import make_grid

# %% [code] {"execution":{"iopub.status.busy":"2024-10-07T18:28:46.509612Z","iopub.execute_input":"2024-10-07T18:28:46.510195Z","iopub.status.idle":"2024-10-07T18:28:46.518863Z","shell.execute_reply.started":"2024-10-07T18:28:46.510152Z","shell.execute_reply":"2024-10-07T18:28:46.517432Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2024-10-07T18:40:50.390312Z","iopub.execute_input":"2024-10-07T18:40:50.390746Z","iopub.status.idle":"2024-10-07T18:40:50.398841Z","shell.execute_reply.started":"2024-10-07T18:40:50.390708Z","shell.execute_reply":"2024-10-07T18:40:50.397435Z"}}
def getBinaryDataLoader(image_size = (224,224), target_class_name = None, path_to_dataset = None, batch_size = 32):
    
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])
    
    dataset = datasets.ImageFolder(root = path_to_dataset, transform = transform)
    
    target_class = target_class_name
    
    target_class_idx = dataset.class_to_idx[target_class]
    
    binary_dataset = BinaryDataset(dataset, target_class_idx)
    binaryDataLoader = DataLoader(binary_dataset, batch_size=batch_size, shuffle = True)
    return binaryDataLoader
    

# %% [code] {"execution":{"iopub.status.busy":"2024-10-07T18:40:50.603756Z","iopub.execute_input":"2024-10-07T18:40:50.604214Z","iopub.status.idle":"2024-10-07T18:40:50.610961Z","shell.execute_reply.started":"2024-10-07T18:40:50.604170Z","shell.execute_reply":"2024-10-07T18:40:50.609828Z"}}
def getAllDataLoader(image_size = (224,224), path_to_dataset = None, batch_size = 32):
    transform = transforms.Compose([
        transforms.Resize(image_size),  # Adjust size based on your model
        transforms.ToTensor(),
    ])
    all_classes_dataset = datasets.ImageFolder(root=path_to_dataset, transform=transform)
    all_classes_dataloader = DataLoader(all_classes_dataset, batch_size=batch_size, shuffle=True)
    return all_classes_dataloader

# %% [code] {"execution":{"iopub.status.busy":"2024-10-07T18:46:33.321131Z","iopub.execute_input":"2024-10-07T18:46:33.321587Z","iopub.status.idle":"2024-10-07T18:46:33.330131Z","shell.execute_reply.started":"2024-10-07T18:46:33.321544Z","shell.execute_reply":"2024-10-07T18:46:33.329002Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2024-10-07T18:45:57.683631Z","iopub.execute_input":"2024-10-07T18:45:57.684088Z","iopub.status.idle":"2024-10-07T18:46:10.314162Z","shell.execute_reply.started":"2024-10-07T18:45:57.684047Z","shell.execute_reply":"2024-10-07T18:46:10.313007Z"}}
if __name__ == '__main__':
    binDL = getBinaryDataLoader(image_size=(224,224),target_class_name="Normal", 
                                path_to_dataset="/kaggle/input/vce-dataset/training", batch_size= 32)
    visualize_batch(binDL)
    
    allDL = getAllDataLoader(image_size=(224,224), path_to_dataset="/kaggle/input/vce-dataset/training", batch_size=32)
    visualize_batch(allDL)

# %% [code]
