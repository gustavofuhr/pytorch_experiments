import torch
import timm



def get_optimizer(model, optimizer_arg, weight_decay = 1e-4):
    if optimizer_arg == "adamp":
        optimizer = timm.optim.AdamP(model.parameters(), lr=0.01, weight_decay=weight_decay)
    elif optimizer_arg == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=weight_decay)
    elif optimizer_arg == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=weight_decay)

    return optimizer
