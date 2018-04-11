import argparse
import pathlib

import chainer
from chainer.training import extensions

import params
from utils import Preprocess
from net import ConvNet

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

files = [str(path) for path in pathlib.Path(params.root).glob('*/*.wav')]

preprocess = Preprocess(params.balance_sentences, params.sr, params.length)
dataset = chainer.datasets.TransformDataset(files, preprocess)
iterator = chainer.iterators.SerialIterator(dataset, params.batchsize)

model = ConvNet(params.n_category)
if args.gpu >= 0:
    chainer.backends.cuda.get_device_from_id(args.gpu).use()
    model.to_gpu()

optimizer = chainer.optimizers.Adam(params.lr)
optimizer.setup(model)
updater = chainer.training.StandardUpdater(
    iterator, optimizer, converter=preprocess.convert, device=args.gpu)
trainer = chainer.training.Trainer(updater, params.finish_trigger)

trainer.extend(extensions.LogReport(trigger=params.report_trigger))
trainer.extend(
    extensions.PrintReport(['iteration', 'main/nll', 'main/likelihood']),
    trigger=params.report_trigger)
trainer.extend(extensions.PlotReport(
    ['main/nll'], 'iteration', file_name='nll.png',
    trigger=params.report_trigger))
trainer.extend(extensions.PlotReport(
    ['main/likelihood'], 'iteration', file_name='likelihood.png',
    trigger=params.report_trigger))
trainer.extend(extensions.ProgressBar(update_interval=5))

trainer.run()
