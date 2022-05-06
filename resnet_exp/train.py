import os
import argparse
import time
import copy
from collections import deque

import numpy as np
import torch
import torch.nn as nn

import models
import dataloaders

def train_model(backbone_name, pretrained, freeze_all, train_loader, val_loader, n_epochs = 100, track_experiment = False):
    """
    Train a model given model params and dataset loaders
    """
    model = models.get_model(backbone_name, pretrained)
    print("model {backbone_name}")
    print(model)
    # the backbone, usually will not restrict the input size of the data
    # e.g.: before the fc of Resnet we have (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
    # but the input channels are related:
    # (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    if freeze_all:
        # make every parameter freeze, fc will be redone and unfreeze
        for param in model.parameters():
            param.requires_grad = False

    # TODO, why it works when the last layer is not resized!?
    no_features_fc = model.fc.in_features
    model.fc = nn.Linear(no_features_fc, len(train_loader.dataset.classes))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:",device)

    model.to(device)
    if track_experiment:
        import wandb
        wandb.watch(model)

    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0


    dataloaders = {
        "train": train_loader,
        "val": val_loader
    }
    cls_nms = val_loader.dataset.classes
    phases = ["train", "val"]
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in phases}
    num_epochs = n_epochs

    start = time.time()
    for epoch in range(num_epochs):
        start_epoch = time.time()
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        if track_experiment:
            epoch_log = {}

        # Each epoch has a training and validation phase
        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_losses = []
            running_corrects = 0.0
            count_corrects = 0

            wrong_epoch_images = deque(maxlen=32)
            wrong_epoch_attr = deque(maxlen=32)

            # Iterate over data.
            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    #print("ouputs", outputs.shape)
                    loss = criterion(outputs, labels)
                    #print("loss", loss)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_losses.append(loss.item())
                running_corrects += torch.sum(preds == labels.data)
                count_corrects += len(preds)

                if phase == "val":
                    wrong_epoch_images.extend([x for x in inputs[preds!=labels]])
                    wrong_epoch_attr.extend([(cls_nms[labels[i]], cls_nms[preds[i]])\
                                                    for i in (preds!=labels).nonzero().flatten()])

            # TODO: not using scheduler yet
            # if phase == 'train':
            #    scheduler.step()
            # Decay LR by a factor of 0.1 every 7 epochs
            # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

            epoch_loss = np.mean(running_losses)
            epoch_acc = 100 * running_corrects.double() / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%')

            if track_experiment:
                epoch_log.update({
                    f"{phase}_loss": epoch_loss,
                    f"{phase}_acc": epoch_acc,
                })
            # deep copy the model
            # if phase == 'val' and epoch_acc > best_acc:
            #    best_acc = epoch_acc
            #    best_model_wts = copy.deepcopy(model.state_dict())

        duration_epoch = time.time() - start_epoch

        if track_experiment:
            epoch_log.update({"duration_epoch": duration_epoch})
            epoch_log.update({"wrong_in_epoch" : [wandb.Image(im, caption=f"GT:{attr[0]} Pred:{attr[1]}")\
                                        for im, attr in zip(wrong_epoch_images, wrong_epoch_attr)]})
            wandb.log(epoch_log)

        print()


    time_elapsed = time.time() - since
    wandb.run.summary["total_duration"] = time_elapsed
    wandb.run.summary["best_val_acc"] = best_acc
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def train(args):
    resize_size = int(args.resize_size) if args.resize_size is not None else None
    train_loader, val_loader = dataloaders.get_dataset_loaders(args.dataset_name,
                                            resize_size, int(args.batch_size))

    if args.track_experiment:
        import wandb
        if args.experiment_group == "" or args.experiment_name == "":
            raise Exception("Should define both the experiment group and name.")
        wandb.init(project=args.experiment_group, name=args.experiment_name, entity=args.wandb_user, )
        wandb.config = args

    train_model(args.backbone, not args.no_transfer_learning, args.freeze_all, train_loader,
                                        val_loader, int(args.n_epochs), args.track_experiment)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="resnet18")


    parser.add_argument("--no_transfer_learning", action=argparse.BooleanOptionalAction)
    parser.add_argument("--freeze_all", action=argparse.BooleanOptionalAction)

    parser.add_argument("--dataset_name", default="CIFAR10")
    parser.add_argument("--resize_size", default=None)

    parser.add_argument("--batch_size", default=32)
    parser.add_argument("--n_epochs", default=50)

    parser.add_argument('--track_experiment', action=argparse.BooleanOptionalAction)
    parser.add_argument("--experiment_group", default="resnet_experiments")
    parser.add_argument("--experiment_name", default="")
    parser.add_argument("--wandb_user", default="gfuhr2")

    args = parser.parse_args()
    train(args)