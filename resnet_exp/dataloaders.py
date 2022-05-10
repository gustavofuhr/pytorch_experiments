import os
import inspect

import torch
import torchvision
import torchvision.transforms as transforms

import image_folder_dataset

CUSTOM_DATASETS = {
    "dogbreeds": "../data/dogbreeds_clean/"
}

def _get_pytorch_dataloders(train_dataset, val_dataset, batch_size):
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)

    return train_loader, val_loader


def _get_pytorch_default_transform(resize_size):

    def_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x)
    ])

    if resize_size is not None:
        def_transform = transforms.Compose([
            transforms.Resize((resize_size,resize_size)),
            def_transform
        ])

    return def_transform


def get_pytorch_dataset_loaders(dataset_name, resize_size, batch_size):
    dataset = getattr(torchvision.datasets, dataset_name)
    dataset_sig = inspect.signature(dataset)

    all_transform = _get_pytorch_default_transform(resize_size)

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


    return _get_pytorch_dataloders(train_dataset, val_dataset, resize_size)

def get_dataset_loaders(dataset_name, resize_size = None, batch_size = 32):

    if dataset_name in dir(torchvision.datasets):
        return get_pytorch_dataset_loaders(dataset_name, resize_size, batch_size)
    elif dataset_name in CUSTOM_DATASETS.keys():
        #custom_train_dataset = image_folder_dataset.ImageFolderDataset(CUSTOM_DATASETS[dataset_name], split="train")
        #custom_val_dataset = image_folder_dataset.ImageFolderDataset(CUSTOM_DATASETS[dataset_name], split="val")

        custom_train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(CUSTOM_DATASETS[dataset_name], "train"),
                                                                            transform=_get_pytorch_default_transform(resize_size))
        custom_val_dataset = torchvision.datasets.ImageFolder(root=os.path.join(CUSTOM_DATASETS[dataset_name], "val"),
                                                                            transform=_get_pytorch_default_transform(resize_size))

        return _get_pytorch_dataloders(custom_train_dataset, custom_val_dataset, batch_size)


if __name__ == "__main__":
    get_dataset_loaders("dogbreeds")

