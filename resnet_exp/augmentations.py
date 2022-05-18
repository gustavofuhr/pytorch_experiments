from torchvision import transforms
from timm.data.transforms_factory import create_transform

DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


def rand_augmentation(resize_size, rand_string_name):
    return create_transform(resize_size, is_training=True, auto_augment=rand_string_name), \
                create_transform(resize_size, is_training=False, auto_augment=rand_string_name)


def _no_augumentation(resize_size):
    return transforms.Compose([
                transforms.Resize(resize_size),
                transforms.CenterCrop(resize_size), #why resize and center_crop are needed
                transforms.ToTensor(),
                transforms.Normalize(DEFAULT_MEAN, DEFAULT_STD)
            ])


def simple_augmentation(resize_size):
    return transforms.Compose([
                transforms.Resize(resize_size),
                transforms.RandomResizedCrop(resize_size, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=(0, 180)),
                transforms.ToTensor(),
                transforms.Normalize(DEFAULT_MEAN, DEFAULT_STD)
            ]), _no_augumentation(resize_size)


def get_augmentations(resize_size, args):
    if args.randaug_string is not None:
        return rand_augmentation(resize_size, args.randaug_string)
    elif args.aug_simple:
        return simple_augmentation(resize_size)
    else:
        return _no_augumentation(resize_size), _no_augumentation(resize_size)