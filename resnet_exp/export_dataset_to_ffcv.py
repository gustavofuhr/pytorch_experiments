import os
import argparse

import torchvision
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField

import image_folder_dataset

def export_dataset_to_ffcv(dataset_name, root_folder, jpeg_quality=90, max_resolution=256):
    #train_dataset = image_folder_dataset.ImageFolderDataset(root_folder, split="train")
    train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(root_folder, "train"))


    write_path = f
    '../data/ffcv/{dataset_name}_train.beton'

    writer = DatasetWriter(write_path, {
        # Tune options to optimize dataset size, throughput at train-time
        'image': RGBImageField(
                max_resolution=max_resolution,
                jpeg_quality=jpeg_quality
            ),
            'label': IntField()
        }
    )

    print(f"Writing dataset {write_path}...")
    writer.from_indexed_dataset(train_dataset)
    print("done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--root_folder", required=True)

    args = parser.parse_args()
    export_dataset_to_ffcv(args.dataset_name, args.root_folder)