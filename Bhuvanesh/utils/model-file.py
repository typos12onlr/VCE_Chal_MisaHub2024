# %% [code]
# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-08T17:33:53.426057Z","iopub.execute_input":"2024-10-08T17:33:53.426506Z","iopub.status.idle":"2024-10-08T17:33:58.296559Z","shell.execute_reply.started":"2024-10-08T17:33:53.426464Z","shell.execute_reply":"2024-10-08T17:33:58.295346Z"}}
import torch
import torchvision.models as models
import torchinfo
from vit_cnn import ViTCNNModel

custom_models = {"vitcnn" : ViTCNNModel}

def getModel(model_name):
    if model_name.lower() in custom_models:
        model = custom_models[model_name.lower()]
    else:
    # Get the model dynamically using getattr
        model_func = getattr(models, model_name.lower())  # Fetch the model constructor by name
        model = model_func(pretrained=True)  # Load the pretrained model
    
    return model

def getList():
    return models.list_models() + list(custom_models.keys())

if __name__ == "__main__":
    classifier = getModel("vitcnn")
    model = classifier()
    print(model)
    




