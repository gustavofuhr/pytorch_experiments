from torchvision import datasets, models, transforms


def get_model(model_name, pretrained = True):
    return getattr(models, model_name)(pretrained=pretrained)