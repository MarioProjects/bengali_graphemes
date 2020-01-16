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
