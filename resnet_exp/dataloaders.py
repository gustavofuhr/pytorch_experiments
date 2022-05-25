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
    "liveness_simple": "/mnt/data/third_p_liveness_2/",
    "flash_ds": "/mnt/data/flash_ds/"
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


class DatasetJoin(torch.utils.data.ConcatDataset):

    def __init__(self, imagefolder_dataset_list):
        super(DatasetJoin, self).__init__(imagefolder_dataset_list)
        self.join_classes()

    def join_classes(self):
        join_class_to_idx = None
        for ds in self.datasets:
            if join_class_to_idx is None:
                join_class_to_idx = ds.class_to_idx
            else:
                target_mapping = {}
                new_classes = []
                for k in ds.classes:
                    if k in join_class_to_idx.keys() and join_class_to_idx[k] != ds.class_to_idx[k]: # different ids
                        # shoud use a transform to the previously define int
                        target_mapping[ds.class_to_idx[k]] = join_class_to_idx[k]
                    else:
                        new_classes.append(k)
                sub_class_to_idx = {key: ds.class_to_idx[key] for key in new_classes}
                join_class_to_idx.update(sub_class_to_idx)
                ds.target_transform = lambda y: target_mapping.get(y, y)
                print("transform in dataset: ", target_mapping)

        self.join_class_to_idx = join_class_to_idx
        s = set().union(*[ds.classes for ds in self.datasets])
        self.classes = list(s)
        #import pdb; pdb.set_trace()


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
        for ith_ds, ds_name in enumerate(dataset_names[s]):
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
        combined_datasets[s] = DatasetJoin(split_datasets)
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
    dataset_names = {
        "train": ["liveness_simple", "flash_ds"],
        "val": ["liveness_simple"]
    }
    from resnet_exp.augmentations import simple_augmentation
    empty_transf = simple_augmentation(128)

    transforms = {
        "train": empty_transf,
        "val": empty_transf
    }
    train_dataloader, val_dataloader = get_dataset_loaders(dataset_names, transforms)
