from fastai.vision import *
from utils.mish_activation import *

# The starter model is based on DenseNet121, which I found to work quite well for such kind of problems.
# The first conv is replaced to accommodate for 1 channel input, and the corresponding pretrained weights are summed.
# ReLU activation in the head is replaced by Mish, which works noticeably better for all tasks I checked it so far.
# Since each portion of the prediction (grapheme_root, vowel_diacritic, and consonant_diacritic)
# is quite independent concept, I create a separate head for each of them
# (though I didn't do a comparison with one head solution).

# change the first conv to accept 1 chanel input
class Dnet_1ch(nn.Module):
    def __init__(self, head, classes=[], pretrained=True, ps=0.5):
        super().__init__()
        m = models.densenet121(pretrained)

        # conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # w = (m.features.conv0.weight.sum(1)).unsqueeze(1)
        # conv.weight = nn.Parameter(w)
        first_conv = m.features.conv0  # conv

        self.layer0 = nn.Sequential(first_conv, m.features.norm0, nn.ReLU(inplace=True))
        self.layer1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
            m.features.denseblock1
        )
        self.layer2 = nn.Sequential(m.features.transition1, m.features.denseblock2)
        self.layer3 = nn.Sequential(m.features.transition2, m.features.denseblock3)
        self.layer4 = nn.Sequential(m.features.transition3, m.features.denseblock4, m.features.norm5)

        nc = self.layer4[-1].weight.shape[0]

        self.heads = []
        for heads_classes in classes:
            self.heads.append(head(nc, heads_classes, ps=ps).cuda())

        #self.head1 = head(nc, classes[0], ps=ps)
        #self.head2 = head(nc, classes[1], ps=ps)
        #self.head3 = head(nc, classes[2], ps=ps)
        ## to_Mish(self.layer0), to_Mish(self.layer1), to_Mish(self.layer2)
        ## to_Mish(self.layer3), to_Mish(self.layer4)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        res = []
        for current_head in self.heads:
            res.append(current_head(x))

        return res

        #x1 = self.head1(x)
        #x2 = self.head2(x)
        #x3 = self.head3(x)

        #return x1, x2, x3