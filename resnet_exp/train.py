import argparse
import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import wandb

import models

def train_model(backbone_name, pretrained, unfreeze_all, train_loader, val_loader):
    """
    Train a model given model params and dataset loaders
    """
    model = models.get_model(backbone_name, pretrained)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:",device)

    model.to(device)
    wandb.watch(model)

    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



def get_dataset_loaders(dataset_name, batch_size = 32):
    dataset = getattr(torchvision.datasets, dataset_name)
    train_dataset = dataset(
        root="../data/",
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )

    val_dataset = dataset(
        root='../data/',
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

    return train_loader, val_loader

def train(args):
    train_loader, val_loader = get_dataset_loaders(args.dataset_name, args.batch_size)

    if args.track_experiment:
        import wandb
        if args.experiment_group == "" or args.experiment_name == "":
            raise Exception("Should define both the experiment group and name.")
        wandb.init(project=args.experiment_group, name=args.experiment_name, entity=args.wandb_user, )
        wandb.config = args

    train_model(args.backbone, args.pretrained, args.unfreeze_all, train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--pretrained", default=True)
    parser.add_argument("--unfreeze_all", default=True)

    parser.add_argument("--dataset_name", default="FashionMNIST")
    parser.add_argument("--batch_size", default=32)

    parser.add_argument("--track_experiment", default=True)
    parser.add_argument("--experiment_group", default="resnet_experiments")
    parser.add_argument("--experiment_name", default="")
    parser.add_argument("--wandb_user", default="gfuhr2")

    args = parser.parse_args()
    train(args)