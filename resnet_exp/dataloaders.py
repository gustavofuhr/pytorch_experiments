import os
import inspect
from collections import defaultdict

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

def _get_pytorch_dataloders(dataset, batch_size, num_workers):
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              pin_memory=True)

    return loader


def _get_pytorch_dataset(dataset_name, split, transform):
    dataset = getattr(torchvision.datasets, dataset_name)
    dataset_sig = inspect.signature(dataset)

    # all_transform = _get_pytorch_default_transform(resize_size)

    if "train" in dataset_sig.parameters:
        dataset = dataset(root="../data/", train=(split == "train"),
                                            transform=transform, download=True)
    elif "split" in dataset_sig.parameters:
        dataset = dataset(root="../data/", split=split,
                                            transform=transform, download=True)
    else:
        raise Exception("Don't understand dataset method signature.")

    return dataset

def _get_image_folder_dataset(dataset_name, split, transform):
    dataset = torchvision.datasets.ImageFolder(root=os.path.join(CUSTOM_DATASETS[dataset_name], split),
                                                                            transform=transform)

    return dataset

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

def get_dataset_loaders(dataset_names,
                            transforms,
                            use_ffcv = False,
                            resize_size = None,
                            batch_size = 32,
                            num_workers = 4):

    """
    Expecting dataset_names and transforms to be dict with "train" and "val" keys
    """
    splits = ["train", "val"]
    combined_datasets = {}
    data_loaders = {}
    for s in splits:
        split_datasets = []
        for ds_name in dataset_names[s]:
            if ds_name in dir(torchvision.datasets):
                this_dataset = _get_pytorch_dataset(ds_name, s, transforms[s])
            elif use_ffcv:
                # TODO; fix this also, if its worth it.
                print("Using FFCV")
                return get_ffcv_dataloaders(CUSTOM_DATASETS[dataset_name], dataset_name, resize_size, batch_size, num_workers)
            elif ds_name in CUSTOM_DATASETS.keys():
                this_dataset = _get_image_folder_dataset(ds_name, s, transforms[s])

            split_datasets.append(this_dataset)

        # TODO: https://stackoverflow.com/questions/71173583/concat-datasets-in-pytorch
        combined_datasets[s] = torch.utils.data.ConcatDataset(split_datasets)
        data_loaders[s] = _get_pytorch_dataloders(combined_datasets[s], batch_size, num_workers)

    return data_loaders["train"], data_loaders["val"]


# def DEPRECATED_get_pytorch_default_transform(resize_size):

#     def_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x)
#     ])

#     if resize_size is not None:
#         def_transform = transforms.Compose([
#             transforms.Resize((resize_size,resize_size)),
#             def_transform
#         ])

#     return def_transform


if __name__ == "__main__":
    get_dataset_loaders("dogbreeds")
