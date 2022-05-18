import os
import argparse
import time
import copy
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import models
import dataloaders
import augmentations
import optimizers
import schedulers
import metrics



def train_model(model,
                train_loader,
                val_loader,
                optimizer,
                scheduler,
                use_ffcv,
                n_epochs = 100,
                metric_eer = False,
                track_experiment = False,
                track_images = False):
    """
    Train a model given model params and dataset loaders
    """
    # import pdb; pdb.set_trace()

    torch.backends.cudnn.benchmark = True

    # the backbone, usually will not restrict the input size of the data
    # e.g.: before the fc of Resnet we have (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
    # but the input channels are related:
    # (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:",device)

    model.to(device)
    if track_experiment:
        import wandb
        wandb.watch(model)

    criterion = nn.CrossEntropyLoss()
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0


    dataloaders = {
        "train": train_loader,
        "val": val_loader
    }

    # TODO: from where to get
    # cls_nms = val_loader.dataset.classes
    phases = ["train", "val"]
    if use_ffcv:
        dataset_sizes = {x: len(dataloaders[x].indices) for x in phases}
    else:
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

            running_loss = 0.0
            running_corrects = 0.0
            
            running_labels = []
            running_outputs = []

            wrong_epoch_images = deque(maxlen=32)
            wrong_epoch_attr = deque(maxlen=32)

            # Iterate over data.
            for batch_idx, (inputs, labels) in enumerate(tqdm(dataloaders[phase])):
                running_labels.append(labels)
                # TODO: needs to cast to float.
                inputs = inputs.float().to(device)
                # TODO: a bunch of stupid convertion for label.
                labels = labels.type(torch.LongTensor).flatten().to(device)

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
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)
                running_outputs.append(outputs.cpu())

                if phase == "val":
                    wrong_epoch_images.extend([x for x in inputs[preds!=labels]])
                    if track_images:
                        wrong_epoch_attr.extend([(labels[i], preds[i])\
                                                    for i in (preds!=labels).nonzero().flatten()])

            if phase == 'train':
                scheduler.step()

            if metric_eer:
                probs = metrics.softmax(torch.cat(running_outputs)).cpu().detach().numpy()
                scores = probs[:,1]
                
                epoch_labels = torch.cat(running_labels)
                epoch_eer = 100 * metrics.eer_metric(epoch_labels, scores)

            epoch_loss = running_loss/len(dataloaders[phase])
            epoch_acc = 100 * running_corrects.double() / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%')
            
            if track_experiment:
                epoch_log.update({
                    f"{phase}_loss": epoch_loss,
                    f"{phase}_acc": epoch_acc,
                })
                if metric_eer:
                    epoch_log.update({
                        f"{phase}_eer": epoch_eer
                    })

        duration_epoch = time.time() - start_epoch

        if track_experiment:
            epoch_log.update({"duration_epoch": duration_epoch})
            if track_images:
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

    train_transform, val_transform = augmentations.get_augmentations(resize_size, args)

    train_loader, val_loader = dataloaders.get_dataset_loaders(args.dataset_name,
                                                                    train_transform,
                                                                    val_transform,
                                                                    args.use_ffcv,
                                                                    resize_size,
                                                                    int(args.batch_size),
                                                                    int(args.num_dataloader_workers))

    model = models.get_model(args.backbone, len(train_loader.dataset.classes),
                                        not args.no_transfer_learning, args.freeze_all_but_last)
    print(f"model {args.backbone}")
    print(model)

    optimizer = optimizers.get_optimizer(model, args.optimizer, args.weight_decay)
    scheduler = schedulers.get_scheduler(optimizer, args)

    if args.track_experiment:
        import wandb
        if args.experiment_group == "" or args.experiment_name == "":
            raise Exception("Should define both the experiment group and name.")
        wandb.init(project=args.experiment_group, name=args.experiment_name, entity=args.wandb_user)
        wandb.config = args

    train_model(model, train_loader, val_loader, optimizer, scheduler, args.use_ffcv,
                    int(args.n_epochs), args.metric_eer, args.track_experiment, args.track_images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="resnet50")


    parser.add_argument("--no_transfer_learning", action=argparse.BooleanOptionalAction)
    parser.add_argument("--freeze_all_but_last", action=argparse.BooleanOptionalAction)

    parser.add_argument("--dataset_name", default="CIFAR10")
    parser.add_argument("--resize_size", default=None)
    parser.add_argument("--use_ffcv", action=argparse.BooleanOptionalAction)
    parser.add_argument("--num_dataloader_workers", default=4) # recomends to be 4 x #GPU

    parser.add_argument("--batch_size", default=64)
    parser.add_argument("--n_epochs", default=50)

    parser.add_argument('--track_experiment', action=argparse.BooleanOptionalAction)
    parser.add_argument("--experiment_group", default="resnet_experiments")
    parser.add_argument("--experiment_name", default="")
    parser.add_argument("--track_images", action=argparse.BooleanOptionalAction)
    parser.add_argument("--wandb_user", default="gfuhr2")

    # lets define some possible augmentations
    parser.add_argument("--randaug_string", default=None)
    parser.add_argument("--aug_simple",  action=argparse.BooleanOptionalAction)

    # options for optimizers
    parser.add_argument("--optimizer", default="sgd") # possible adam, adamp and sgd
    parser.add_argument("--weight_decay", default=1e-4)

    # options for liveness
    parser.add_argument("--metric_eer", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    train(args)
