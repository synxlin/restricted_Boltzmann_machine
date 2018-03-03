from __future__ import print_function

import os
import time
import argparse
import numpy as np

import rbm
from utils import ReadDigitImageData


parser = argparse.ArgumentParser(description='Digits Recognition using RBM')
parser.add_argument('-m', '--method', default=0, type=int, dest='method',
                    help="0: net | 1: forest | 2: rbm")
parser.add_argument('-i', '--visible-label', default=0, type=int, dest='visible_label',
                    help="0: label is output | 1: label is visible units")
parser.add_argument('-n', '--num-layers', default=3, type=int, dest='num_layers',
                    help="number of RBM net layers")
parser.add_argument('-d', '--dim-hidden', default='784, 500, 500, 2000', dest='dim_hidden',
                    help="dim of hidden units, format as 1,2,3")
parser.add_argument('-o', '--output-dim', default=10, type=int, dest='output_dim',
                    help="dim of output label: 10 | 4")
parser.add_argument('--lr', default=0.1, type=float, dest='lr',
                    help="learning rate")
parser.add_argument('-b', default=100, type=int, dest='batch_size',
                    help="batch size")
parser.add_argument('--max-epoch-rbm', default=50, type=int, dest='max_epochs_rbm',
                    help="max epoch for training RBM only")
parser.add_argument('--max-epoch-joint', default=200, type=int, dest='max_epochs_joint',
                    help="max epoch for joint-training RBM net")
parser.add_argument('--path-mnist', default=None, dest='mnist_path', help="path to mnist data")
parser.add_argument('--path', default=None, dest='path', help="path to image data")
parser.add_argument('--noise-level', default=0, type=int, dest='noise_level', help="noise_level")

if __name__ == '__main__':
    args = parser.parse_args()

    method = args.method
    visible_label = args.visible_label

    output_dim = args.output_dim
    num_layers = args.num_layers
    dim = list(map(lambda x: int(x), args.dim_hidden.split(',')))
    if method == 0:
        assert num_layers == len(dim) - 1 and dim[0] == 784
    elif method == 1:
        assert len(dim) == 1 and dim[0] == 784 and output_dim == 10
        visible_label = 0
    else:
        assert len(dim) == 2 and dim[0] == 784 + output_dim
        visible_label = 1

    learning_rate = args.lr
    batch_size = args.batch_size
    max_epochs_rbm = args.max_epochs_rbm
    max_epochs_joint = args.max_epochs_joint

    read_output_type = 0
    if method == 1:
        read_output_type = -1
    elif method == 2:
        read_output_type = 1

    # read image data
    assert args.mnist_path is not None
    rd = ReadDigitImageData(dim=output_dim)
    train_set, train_label_set = rd.read(path=args.mnist_path, trainflag=1, mnistflag=1, output_type=read_output_type)
    test_set, test_label_set = rd.read(path=args.mnist_path, trainflag=0, mnistflag=1, output_type=read_output_type)
    if method == 0 or method == 2:
        train_set, train_label_set = rd.make_batches(train_data=train_set, train_label=train_label_set,
                                                     batch_size=batch_size)
    if method == 0:
        test_set, test_label_set = rd.make_batches(train_data=test_set, train_label=test_label_set,
                                                   batch_size=batch_size)
    elif method == 2:
        test_set = [np.array(test_set)]

    test_set = [test_set]
    test_label_set = [test_label_set]
    test_name_set = ['MNIST_test']

    if args.path is not None:
        testset, labelset = rd.read(path=args.path, trainflag=0, mnistflag=0, output_type=read_output_type,
                                    noise_level=args.noise_level)
        if len(testset) > 0:
            if method == 0 or method == 2:
                testset = [np.array(testset)]
                if method == 0:
                    labelset = [labelset]
            test_set.append(testset)
            test_label_set.append(labelset)
            test_name = 'Test' + ('_%d' % args.noise_level)
            test_name_set.append(test_name)
            print("read test set %d / level %d done" % (args.path, args.noise_level))

    path = os.path.join('exp', 'method_%d_%s' % (method, time.strftime('%H_%M_%S', time.gmtime())))
    if not os.path.exists(path):
        os.mkdir(path)
    path_log = os.path.join(path, 'hyper_param.log')
    with open(path_log, 'wt') as f:
        f.write('method {0} \nvisible_label {9}\noutput_dim {1}\nnum_layers {2}\ndim {3}\n'
                'learning_rate {4}\nbatch_size {5}\nmax_epochs_rbm {6}\nmax_epochs_joint {7}\n'
                'read_output_type {8}'.format(args.method, args.output_dim, args.num_layers,
                                              dim, args.lr, args.batch_size, args.max_epochs_rbm,
                                              args.max_epochs_joint, read_output_type, visible_label))

    if method == 0:
        r = rbm.RbmNet(num_layers=num_layers, dim=dim, learning_rate=learning_rate, path=path, mode=visible_label)
        r.train_rbms(batch_data=train_set, batch_label=train_label_set, max_epochs_rbm=max_epochs_rbm,
                     max_epochs_joint=max_epochs_joint, test_set=test_set, test_label_set=test_label_set,
                     test_name_set=test_name_set)
        print("Train RBM-Net Successfully")
        for i in range(0, len(test_set)):
            r.predict(batch_test=test_set[i], batch_test_label=test_label_set[i], test_name=test_name_set[i])

    elif method == 1:
        r = rbm.RbmForest(num_visible=dim[0], num_hidden=dim[1], num_output=output_dim,
                          learning_rate=learning_rate, path=path)

        r.train(train_data=train_set, batch_size=batch_size, max_epochs=max_epochs_rbm)

        test_num = len(test_set)
        for i in range(0, test_num):
            ans = r.predict(test_set[i])
            v = list(map(lambda x: x[0] - x[1], zip(ans, test_label_set[i])))
            err = 1 - v.count(0) / len(v)
            datafile = os.path.join(path, 'final_test.txt')
            for j in range(0, ans.__len__()):
                ans[j] = str(ans[j])
            str_convert = ''.join(ans)
            with open(datafile, 'at') as f:
                f.write("\n%s: error: %s\n" % (test_name_set[i], err))
                f.write('%s\n' % str_convert)

    else:
        r = rbm.RBM(num_visible=dim[0], num_hidden=dim[1], learning_rate=learning_rate, path=path)
        for epoch in range(0, max_epochs_rbm):
            datafile = os.path.join(path, 'final_test.txt')
            with open(datafile, 'at') as f:
                f.write('\n epoch: %d \n' % epoch)
            r.train(train_set, max_epochs=1)

            test_num = len(test_set)
            for i in range(0, test_num):
                ans = r.predict(test_set[i], soft_max=dim)
                ans = ans[0]
                v = list(map(lambda x: x[0] - x[1], zip(ans, test_label_set[i])))
                err = 1 - v.count(0) / len(v)
                with open(datafile, 'at') as f:
                    f.write(" %s: error: %s\n" % (test_name_set[i], err))
