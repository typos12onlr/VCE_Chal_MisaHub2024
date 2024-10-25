# %% [code] {"execution":{"iopub.status.busy":"2024-10-24T15:53:55.889309Z","iopub.execute_input":"2024-10-24T15:53:55.890154Z","iopub.status.idle":"2024-10-24T15:54:09.225209Z","shell.execute_reply.started":"2024-10-24T15:53:55.890115Z","shell.execute_reply":"2024-10-24T15:54:09.224044Z"}}
# !pip install vit-pytorch

# %% [code] {"execution":{"iopub.status.busy":"2024-10-24T15:54:23.158333Z","iopub.execute_input":"2024-10-24T15:54:23.158722Z","iopub.status.idle":"2024-10-24T15:54:27.977312Z","shell.execute_reply.started":"2024-10-24T15:54:23.158685Z","shell.execute_reply":"2024-10-24T15:54:27.976496Z"}}
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torchvision.models as models
from torch.utils.data import DataLoader
import torchinfo

from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# %% [code] {"execution":{"iopub.status.busy":"2024-10-24T15:54:41.022658Z","iopub.execute_input":"2024-10-24T15:54:41.023030Z","iopub.status.idle":"2024-10-24T15:54:41.030043Z","shell.execute_reply.started":"2024-10-24T15:54:41.022996Z","shell.execute_reply":"2024-10-24T15:54:41.029013Z"}}
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_names = sorted(os.listdir(image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_names[idx])
        image = Image.open(img_name)  # Load the image and convert to RGB
        if self.transform:
            image = self.transform(image)
        return image, self.image_names[idx]  # Return image and its filename


# %% [code] {"execution":{"iopub.status.busy":"2024-10-24T16:10:14.041025Z","iopub.execute_input":"2024-10-24T16:10:14.041849Z","iopub.status.idle":"2024-10-24T16:10:14.053739Z","shell.execute_reply.started":"2024-10-24T16:10:14.041806Z","shell.execute_reply":"2024-10-24T16:10:14.052724Z"}}
def run_inference(model, dataloader, device, existing_label_class_mapping, actual_class_order, output_dir):
    model.eval()  # Set the model to evaluation mode
    model= model.to(device)
    predictions = {}
    softmax = nn.Softmax()
    with torch.no_grad():  # Disable gradient calculation
        for images, image_names in dataloader:
            images = images.to(device)  # Move images to the same device as model
            outputs = model(images)
            outputs= softmax(outputs)
#             print(outputs.shape)
            _, preds = torch.max(outputs, 1)  # Get the predicted class
#             print(_)
            # Store predictions along with the image names
            for img_name, pred, out in zip(image_names, preds, outputs):
                predictions[img_name] = [pred.item(), out]  # Store prediction as int
                
    # Create the mapping from current class index to the new order
    reorder_map = [list(existing_label_class_mapping.keys())[list(existing_label_class_mapping.values()).index(class_name)] for class_name in actual_class_order]

    # Reverse map: from old index to new index
    reverse_map = {old_index: reorder_map.index(old_index) for old_index in existing_label_class_mapping.keys()}

    # Example of how the predictions are structured
    # predictions = {
    #    'worm1_65.jpg': [9, tensor([...], device='cuda:0')],
    #    'image2.jpg': [2, tensor([...], device='cuda:0')]
    # }

    # Loop over each entry in the predictions dictionary
    for filename, (class_label, prob_tensor) in predictions.items():
        # Update the class label according to the new index
        new_class_label = reverse_map[class_label]

        # Reorder the probabilities using the reorder_map
        reordered_probabilities = prob_tensor[reorder_map]

        # Update the predictions dictionary with the new class label and reordered probabilities
        predictions[filename] = [new_class_label, reordered_probabilities]
        

    rows = []

    # Loop through the predictions and convert each entry to the desired format
    for image_path, (class_label, prob_scores) in predictions.items():
        # Get the predicted class name
        predicted_class = actual_class_order[class_label]

        # Create a row with the image path, probability scores, and predicted class
        row = [image_path] + prob_scores.detach().cpu().numpy().tolist() + [predicted_class]

        # Append the row to the list
        rows.append(row)

    # Create a DataFrame with the appropriate column names
    df = pd.DataFrame(rows, columns=['image_path'] + actual_class_order + ['predicted_class'])

    # Save the DataFrame to an Excel file
    output_filepath = output_dir+'output.xlsx'
    df.to_excel(output_filepath, index=False)
    
    return predictions


if __name__ == '__main__':

    
    model = getModel("resnet50")
    model.fc= nn.Linear(2048, 11)
    checkpoint = torch.load("/kaggle/input/one-at-a-time/pytorch/default/2/resnet50-one-at-a-time.pth", map_location='cpu')
    model.load_state_dict(checkpoint["model_state_dict"])

    # Define the image transforms (resizing and normalization)
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to the input size expected by your model (224x224)
        transforms.ToTensor()
    ])


    # Create the test dataset and dataloader
    test_dir = '/kaggle/input/vce-test'  # Path to the test directory
    test_dataset = TestDataset(image_dir=test_dir, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # Use appropriate batch size


    existing_label_class_map = {0: "Normal", 1: "Erosion", 2: "Polyp", 3: "Angioectasia", 4: "Bleeding", 
                 5: "Lymphangiectasia", 6: "Foreign Body", 7: "Erythema", 8: "Ulcer", 9: "Worms"}
    actual_order = ['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 
                           'Foreign Body', 'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']
    out_dir= "/kaggle/working/"

    
    device= 'cuda' if torch.cuda.is_available() else 'cpu'

    predictions = run_inference(model, test_loader, device, existing_label_class_map, actual_order, out_dir)