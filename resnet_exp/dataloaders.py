import inspect

import torch
import torchvision
import torchvision.transforms as transforms



def get_dataset_loaders(dataset_name, resize_size = None, batch_size = 32):
    dataset = getattr(torchvision.datasets, dataset_name)
    dataset_sig = inspect.signature(dataset)

    all_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x)
        ])

    if resize_size is not None:
        all_transform = transforms.Compose([
            transforms.Resize((resize_size,resize_size)),
            all_transform
        ])

    if "train" in dataset_sig.parameters:
        train_dataset = dataset(root="../data/", train=True,
                                            transform=all_transform, download=True)

        val_dataset = dataset(root='../data/', train=False,
                                            transform=all_transform, download=True)
    elif "split" in dataset_sig.parameters:
        train_dataset = dataset(root="../data/", split="train",
                                            transform=all_transform, download=True)

        val_dataset = dataset(root='../data/', split="test",
                                            transform=all_transform, download=True)
    else:
        raise Exception("Don't understand dataset method signature.")

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)

    return train_loader, val_loader