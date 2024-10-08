# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-10-08T17:33:53.426057Z","iopub.execute_input":"2024-10-08T17:33:53.426506Z","iopub.status.idle":"2024-10-08T17:33:58.296559Z","shell.execute_reply.started":"2024-10-08T17:33:53.426464Z","shell.execute_reply":"2024-10-08T17:33:58.295346Z"}}
import torch
import torchvision.models as models

def getModel(model_name):
    # Get the model dynamically using getattr
    model_func = getattr(models, model_name.lower())  # Fetch the model constructor by name
    model = model_func(pretrained=True)  # Load the pretrained model
    
    return model

def getList():
    return torchvision.models.list_models()

