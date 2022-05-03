import argparse

import torch
import torchvision
import torchvision.transforms as transforms



def train_model(backbone_name, pretrained, unfreeze_all, train_loader, val_loader):
    """
    Train a model given model params and dataset loaders
    """


def get_dataset_loaders(dataset_name, batch_size = 32):
    dataset = torchvision.datasets.getattr(dataset_name)
    train_dataset = dataset(
        root="data/",
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )

    val_dataset = dataset(
        root='data/',
        train=False,
        transform=transforms.ToTensor(),
        download=True
    )

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)

def train(args):
    train_loader, val_loader = get_dataset_loaders(args.dataset_name, args.batch_size)
    train_model(args.backbone, args.pretrained, args.unfreeze_all, train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--pretrained", default=True)
    parser.add_argument("--unfreeze_all", default=True)

    parser.add_argument("--dataset_name", default="FashionMNIST")
    parser.add_argument("--batch_size", default=32)

    args = parser.parse_args()