from .densenet import *
from .seresnext import *
from .resnet import *
from .efficientnet import *
from .heads import *


def model_selector(model_name, head_name, classes, pretrained=True, banglalekha=False):
    """
    :param model_name:
    """

    if "initial_head" in head_name:
        head = InitialHead
    else:
        assert False, "Unknown head selected: {}".format(head_name)

    ps = 0.5
    if "noDrop" in head_name:
        ps = 0.0

    if "densenet121" in model_name:
        model = Dnet_1ch(head, classes=classes, ps=ps, pretrained=pretrained)
        return model
    elif "magicseresnext101" in model_name:
        model = MagicSeresnext101(head, classes=classes, ps=ps, pretrained=pretrained)
        return model
    elif "seresnext101" in model_name:
        if banglalekha: model = Seresnext101Banglalekha(head, classes=classes, ps=ps, pretrained=pretrained)
        else: model = Seresnext101(head, classes=classes, ps=ps, pretrained=pretrained)
        return model
    elif "resnet34" in model_name:
        model = Resnet34(head, classes=classes, ps=ps, pretrained=pretrained)
        return model
    elif "efficientnetb3b" in model_name:
        model = EfficientNetB3b(head, classes=classes, ps=ps)
        return model
    elif "efficientnetb4b" in model_name:
        model = EfficientNetB4b(head, classes=classes, ps=ps)
        return model
    elif "efficientnetb7b" in model_name:
        model = EfficientNetB7b(head, classes=classes, ps=ps)
        return model
    else:
        assert False, "Unknown model selected: {}".format(model_name)
