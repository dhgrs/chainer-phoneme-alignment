import chainer
import chainer.links as L
import chainer.functions as F


class ConvNet(chainer.Chain):

    def __init__(self, n_category):
        super(ConvNet, self).__init__()
        self.n_category = n_category
        with self.init_scope():
            self.conv1 = L.ConvolutionND(1, 1, 32, 4, 2, 1)
            self.conv2 = L.ConvolutionND(1, 32, 32, 4, 2, 1)
            self.conv3 = L.ConvolutionND(1, 32, 32, 4, 2, 1)
            self.conv4 = L.ConvolutionND(1, 32, 32, 4, 2, 1)
            self.conv5 = L.ConvolutionND(1, 32, 32, 4, 2, 1)
            self.conv6 = L.ConvolutionND(1, 32, n_category, 4, 2, 1)

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
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        y = self.conv6(h)
        return y
