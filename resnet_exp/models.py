import timm
from torchvision import models



def get_model(model_name, n_classes, pretrained = True, freeze_all_but_last = False):
    def _freeze_all(model):
        # make every parameter freeze, fc will be redone and unfreeze
        for param in model.parameters():
            param.requires_grad = False

    # always priorize timm if possible
    if model_name in timm.list_models("*"):
        print("Getting model from timm")
        model = timm.create_model(model_name, pretrained, num_classes=n_classes)
        if freeze_all_but_last: _freeze_all(model)
    else:
        print("Getting model from torchvision")
        model = getattr(models, model_name)(pretrained=pretrained)
        if freeze_all_but_last: _freeze_all(model)

        # TODO, why it works when the last layer is not resized!?
        no_features_fc = model.fc.in_features
        if use_ffcv:
            #TODO: debug
            model.fc = nn.Linear(no_features_fc, 100)
        else:
            model.fc = nn.Linear(no_features_fc, n_classes)


    return model