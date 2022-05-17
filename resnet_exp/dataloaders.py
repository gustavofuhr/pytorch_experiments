import os
import inspect

import torch
import torchvision
import torchvision.transforms as transforms

from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, Cutout
from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder


CUSTOM_DATASETS = {
    "dogbreeds": "../data/dogbreeds_clean/",
    "ffcv_dogbreeds": "../data/ffcv/",
    "unico130k_v2": "/home/gfuhr/data/unico130k_v2/ensemble/",
    "liveness_simple": "/mnt/data/third_p_liveness_2/"
}

def _get_pytorch_dataloders(train_dataset, val_dataset, batch_size, num_workers):
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              pin_memory=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              pin_memory=True)

    return train_loader, val_loader


def _get_pytorch_dataset(dataset_name, train_transform, val_transform):
    dataset = getattr(torchvision.datasets, dataset_name)
    dataset_sig = inspect.signature(dataset)

    # all_transform = _get_pytorch_default_transform(resize_size)

    if "train" in dataset_sig.parameters:
        train_dataset = dataset(root="../data/", train=True,
                                            transform=train_transform, download=True)

        val_dataset = dataset(root='../data/', train=False,
                                            transform=val_transform, download=True)
    elif "split" in dataset_sig.parameters:
        train_dataset = dataset(root="../data/", split="train",
                                            transform=train_transform, download=True)

        val_dataset = dataset(root='../data/', split="test",
                                            transform=val_transform, download=True)
    else:
        raise Exception("Don't understand dataset method signature.")


    return train_dataset, val_dataset

def _get_image_folder_dataset(dataset_name, train_transform, val_transform):
    train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(CUSTOM_DATASETS[dataset_name], "train"),
                                                                            transform=train_transform)
    val_dataset = torchvision.datasets.ImageFolder(root=os.path.join(CUSTOM_DATASETS[dataset_name], "val"),
                                                                            transform=val_transform)

    return train_dataset, val_dataset

def get_ffcv_dataloaders(root_dir, dataset_name, resize_size, batch_size, num_workers):
    # Random resized crop
    decoder = RandomResizedCropRGBImageDecoder((resize_size, resize_size))

    # TODO: pipelines should be different for train, val, normally.
    # TODO: augmentation should be done by another lib and equal for testing purposes.
    # Data decoding and augmentation
    image_pipeline = [decoder, ToTensor(), ToTorchImage(), ToDevice(0)]
    label_pipeline = [IntDecoder(), ToTensor(), ToDevice(0)]

    # Pipeline for each data field
    pipelines = {
        'image': image_pipeline,
        'label': label_pipeline
    }

    train_loader = Loader(os.path.join(root_dir, f"{dataset_name}_train.beton"),
                                            batch_size=batch_size, num_workers=num_workers,
                                            order=OrderOption.RANDOM, pipelines=pipelines, os_cache=True)

    val_loader = Loader(os.path.join(root_dir, f"{dataset_name}_val.beton"),
                                            batch_size=batch_size, num_workers=num_workers,
                                            order=OrderOption.RANDOM, pipelines=pipelines, os_cache=True)


    return train_loader, val_loader

def get_dataset_loaders(dataset_name,
                            train_transform,
                            val_transform,
                            use_ffcv = False,
                            resize_size = None,
                            batch_size = 32,
                            num_workers = 4):

    if dataset_name in dir(torchvision.datasets):
        train_dataset, val_dataset = _get_pytorch_dataset(dataset_name, train_transform, val_transform)
    elif use_ffcv:
        print("Using FFCV")
        return get_ffcv_dataloaders(CUSTOM_DATASETS[dataset_name], dataset_name, resize_size, batch_size, num_workers)
    elif dataset_name in CUSTOM_DATASETS.keys():
        train_dataset, val_dataset = _get_image_folder_dataset(dataset_name, train_transform, val_transform)

    return _get_pytorch_dataloders(train_dataset, val_dataset, batch_size, num_workers)


def DEPRECATED_get_pytorch_default_transform(resize_size):

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


if __name__ == "__main__":
    get_dataset_loaders("dogbreeds")
