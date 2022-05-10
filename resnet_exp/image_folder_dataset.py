#from torch.utils.data import Dataset, DataLoader

# Your custom dataset should inherit Dataset and override the following methods:
#     __len__ so that len(dataset) returns the size of the dataset.
#     __getitem__ to support the indexing such that dataset[i] can be used to get iith sample.

import os

import cv2
import torch

import torchvision.transforms as transforms

class ImageFolderDataset(torch.utils.data.Dataset):
    """Custom ImageFolder Dataset."""

    def __init__(self, root, split = "train"):
        """
            root (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.base_path = os.path.join(root, split)

        # create image list and labels
        self.images_fns, self.targets = [], []

        all_subfolders = sorted([f for f in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, f))])
        self.class_to_idx = {f:idx for idx,f in enumerate(all_subfolders)}
        self.classes = list(self.class_to_idx.keys())
        print("self.classes", self.classes)

        for ic, c in enumerate(self.classes):
            IMAGE_EXTENSIONS = (".png", ".jpeg", ".jpg", ".bmp")
            class_images_fns = [os.path.join(self.base_path, c, f) for f in os.listdir(os.path.join(self.base_path, c))\
                                                                                    if f.lower().endswith(IMAGE_EXTENSIONS)]

            # TODO: should I create a data ?
            self.images_fns.extend(class_images_fns)
            self.targets.extend([self.class_to_idx[c]]*len(class_images_fns))

        self.targets = torch.LongTensor(self.targets)
        self.img_dim = (416, 416)

    def __len__(self):
        return len(self.images_fns)

    def __getitem__(self, idx):
        """
        return an image and label given an index.

        NOTE: images are RGB float tensors from 0 to 1.
        TODO: I hate that all these transformations are written here.
        """
        # no ideia why I need this, when the idx would be a tensor!?
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.images_fns[idx]
        image = cv2.imread(img_name)[...,::-1]
        # NOTE: I need to convert to the same size, otherwise I get an error on the DataLoader:
        # RuntimeError: stack expects each tensor to be equal size, but got [480, 640, 3] at entry 0 and [1280, 720, 3] at entry 4
        image = cv2.resize(image, self.img_dim)
        class_id = self.targets[idx]
        # class_id = torch.tensor([class_id])

        img_tensor = torch.from_numpy(image/255.)
        img_tensor = img_tensor.permute(2, 0, 1)

        return img_tensor.float(), class_id


