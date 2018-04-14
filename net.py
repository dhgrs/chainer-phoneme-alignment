import chainer
import chainer.links as L
import chainer.functions as F


class BottleneckA(chainer.Chain):

    def __init__(self, in_channels, mid_channels, out_channels, stride=2):
        super(BottleneckA, self).__init__()
        with self.init_scope():
            self.conv1 = L.ConvolutionND(
                1, in_channels, mid_channels, 1, stride, 0, nobias=True)
            self.bn1 = L.BatchNormalization(mid_channels)
            self.conv2 = L.ConvolutionND(
                1, mid_channels, mid_channels, 9, 1, 4, nobias=True)
            self.bn2 = L.BatchNormalization(mid_channels)
            self.conv3 = L.ConvolutionND(
                1, mid_channels, out_channels, 1, 1, 0, nobias=True)
            self.bn3 = L.BatchNormalization(out_channels)
            self.conv4 = L.ConvolutionND(
                1, in_channels, out_channels, 1, stride, 0, nobias=True)
            self.bn4 = L.BatchNormalization(out_channels)

    def __call__(self, x):
        h1 = F.relu(self.bn1(self.conv1(x)))
        h1 = F.relu(self.bn2(self.conv2(h1)))
        h1 = self.bn3(self.conv3(h1))
        h2 = self.bn4(self.conv4(x))
        return F.relu(h1 + h2)


class BottleneckB(chainer.Chain):

    def __init__(self, in_channels, mid_channels):
        super(BottleneckB, self).__init__()
        with self.init_scope():
            self.conv1 = L.ConvolutionND(
                1, in_channels, mid_channels, 1, 1, 0, nobias=True)
            self.bn1 = L.BatchNormalization(mid_channels)
            self.conv2 = L.ConvolutionND(
                1, mid_channels, mid_channels, 9, 1, 4, nobias=True)
            self.bn2 = L.BatchNormalization(mid_channels)
            self.conv3 = L.ConvolutionND(
                1, mid_channels, in_channels, 1, 1, 0, nobias=True)
            self.bn3 = L.BatchNormalization(in_channels)

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))
        return F.relu(h + x)


class BuildingBlock(chainer.Chain):

    def __init__(self, n_layer, in_channels, mid_channels, out_channels,
                 stride):
        super(BuildingBlock, self).__init__()
        with self.init_scope():
            self.a = BottleneckA(
                in_channels, mid_channels, out_channels, stride)
            self._forward = ['a']
            for i in range(n_layer - 1):
                name = 'b{}'.format(i + 1)
                bottleneck = BottleneckB(out_channels, mid_channels)
                setattr(self, name, bottleneck)
                self._forward.append(name)

    def __call__(self, x):
        for name in self._forward:
            l = getattr(self, name)
            x = l(x)
        return x


class ConvNet(chainer.Chain):

    def __init__(self, n_category):
        super(ConvNet, self).__init__()
        self.n_category = n_category
        blocks = [3, 4, 6, 3]
        with self.init_scope():
            self.conv1 = L.ConvolutionND(1, 1, 64, 48, 2, 23)
            self.bn1 = L.BatchNormalization(64)
            self.res2 = BuildingBlock(blocks[0], 64, 64, 256, 1)
            self.res3 = BuildingBlock(blocks[1], 256, 128, 512, 2)
            self.res4 = BuildingBlock(blocks[2], 512, 256, 1024, 2)
            self.res5 = BuildingBlock(blocks[3], 1024, 512, 2048, 2)
            self.fc6 = L.ConvolutionND(1, 2048, n_category, 1)

    def __call__(self, x, phonemes, lengths):
        y = self.forward(x)

        # The input of ctc must be list or tuple.
        ys = [y[:, :, i] for i in range(y.shape[2])]

        # The input label of ctc must be variable or array.
        phonemes = F.pad_sequence(phonemes, padding=self.n_category-1)

        nll = F.connectionist_temporal_classification(
            ys, phonemes, blank_symbol=self.n_category-1, label_length=lengths)
        likelihood = F.exp(-nll)

        chainer.reporter.report({'nll': nll, 'likelihood': likelihood}, self)
        return nll

    def forward(self, x):
        h1 = F.relu(self.bn1(self.conv1(x)))
        h1 = F.max_pooling_nd(h1, 8, 2, 3)
        h2 = self.res2(h1)
        h3 = self.res3(h2)
        h4 = self.res4(h3)
        h5 = self.res5(h4)
        y = self.fc6(h5)
        return y
