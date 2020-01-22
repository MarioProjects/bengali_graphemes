from pytorchcv.model_provider import get_model as ptcv_get_model
from utils.mish_activation import *

# change the first conv to accept 1 chanel input
class Seresnext101(nn.Module):
    def __init__(self, head, classes=[], ps=0.5, pretrained=True):
        super().__init__()

        model = ptcv_get_model("seresnext101_32x4d", pretrained=pretrained)

        # convBN = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # convBN_weight = model.features.init_block.conv.conv.weight.sum(1).unsqueeze(1)
        # convBN.weight = nn.Parameter(convBN_weight)
        #
        # model.features.init_block.conv.conv = convBN

        self.features = nn.Sequential(*list(model.features.children())[:-1])

        nc = 2048

        self.head1 = head(nc, classes[0], ps=ps)
        self.head2 = head(nc, classes[1], ps=ps)
        self.head3 = head(nc, classes[2], ps=ps)
        ## to_Mish(self.layer0), to_Mish(self.layer1), to_Mish(self.layer2)
        ## to_Mish(self.layer3), to_Mish(self.layer4)

    def forward(self, x):
        x = self.features(x)

        x1 = self.head1(x)
        x2 = self.head2(x)
        x3 = self.head3(x)

        return x1, x2, x3


class MagicSeresnext101(nn.Module):
    def __init__(self, head, classes=[], ps=0.5, pretrained=True):
        super().__init__()

        model = ptcv_get_model("seresnext101_32x4d", pretrained=pretrained)

        modules = {}
        for name, module in model.named_modules():
            if (isinstance(module, nn.Conv2d)):
                stride = module.stride
                if stride == (2, 2) or stride == 2:
                    module.stride = (1, 1)
                    modules[name] = module
                elif stride == 2:
                    module.stride = 1
                    modules[name] = module

        for name in modules:
            parent_module = model
            objs = name.split(".")
            if len(objs) == 1:
                # model.__setattr__(name, modules[name])
                model.__setattr__("magicMaxPool", nn.MaxPool2d(kernel_size=2, stride=2))
                continue

            for obj in objs[:-1]:
                parent_module = parent_module.__getattr__(obj)

            # parent_module.__setattr__(objs[-1], modules[name])
            parent_module.__setattr__("magicMaxPool", nn.MaxPool2d(kernel_size=2, stride=2))

        self.features = nn.Sequential(*list(model.features.children())[:-1])

        nc = 2048

        self.head1 = head(nc, classes[0], ps=ps)
        self.head2 = head(nc, classes[1], ps=ps)
        self.head3 = head(nc, classes[2], ps=ps)
        ## to_Mish(self.layer0), to_Mish(self.layer1), to_Mish(self.layer2)
        ## to_Mish(self.layer3), to_Mish(self.layer4)

    def forward(self, x):
        x = self.features(x)

        x1 = self.head1(x)
        x2 = self.head2(x)
        x3 = self.head3(x)

        return x1, x2, x3

class Seresnext101Banglalekha(nn.Module):
    def __init__(self, head, classes=[84], ps=0.5, pretrained=True):
        super().__init__()

        model = ptcv_get_model("seresnext101_32x4d", pretrained=pretrained)

        convBN = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        convBN_weight = model.features.init_block.conv.conv.weight.sum(1).unsqueeze(1)
        convBN.weight = nn.Parameter(convBN_weight)

        model.features.init_block.conv.conv = convBN

        self.features = nn.Sequential(*list(model.features.children())[:-1])

        nc = 2048

        self.heads = []
        for heads_classes in classes:
            self.heads.append(head(nc, heads_classes, ps=ps).cuda())

    def forward(self, x):
        x = self.features(x)

        res = []
        for current_head in self.heads:
            res.append(current_head(x))

        return res
