# -*-coding:utf-8-*-
from __future__ import print_function

import numpy as np
import pickle
import os

from .rbm import RBM

INT = 0.1
EXT = 3.0
MAX = 20
RATIO = 10
SIG = 0.1
RHO = SIG / 2.0


# Restricted Boltzmann Machine Network
class RbmNet:
    def __init__(self, num_layers=3, dim=None, output_dim=10, learning_rate=0.1, path=None, mode=0):
        """
        初始化函数
        Initial Function
        :param num_layers:  RBM的层数（个数），RBM级联，即上一个RBM的 hidden units 为下个RBM的 visible units
                         the num_layers or numbers of RBM
                         the hidden units of the former RBMis the visible units of the latter RBM
        :param dim:     每个RBM的可见层和隐含层单元数，类型为list
                         第i个元素为第i个RBM可见单元个数，第i+1个元素为第i个RBM隐含单元个数
                         the visible units number and hidden units number of each RBM, type: list
                         the i-th elements of list is the the visible units number of the i-th RBM
                         the i+1-th elements of list is the the hidden units number of the i-th RBM
        :param output_dim: mode 0下RMB Net输出的维度，即通过num_layers层RBM后，通过转移矩阵后输出的维度
                           mode 1下RMB Net标签的维度，即最后一层RBM输入增加的维度数
                            RBM Net ouput dim under mode = 0 (after num_layers RBM and transfer matrix)
                            Extended visible units number of last layer RBM for labeling under mode = 1
        :param learning_rate: RBM的学习率, learning rate of RBM
        :param path:    该RBM网络的参数存储路径
                         the path where we store the parameters of RBM net
        :param mode:    该RBM net的模式，0 代表标签不作为可见单元  1 代表 标签作为可见单元
                         label is used as visible units under mode 1; otherwise, mode 0
        RbmNet 类成员
        w_class:    mode 0 下，从最后一层隐单元转移至标签输出的权重矩阵
                    （1 + num_visible of the top RBM) * the dimension of label(softmax)
                    the weight matrix between the hidden units of the top RBM and the label(softmax)
        """
        self.num_layers = num_layers
        if dim is None:
            self.dim = [788, 500, 500, 2000]
        elif isinstance(dim, list):
            self.dim = dim
        else:
            self.dim = [dim, 500, 500, 2000]
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.path = path
        self.rbms = []
        self.mode = mode
        for i in range(0, num_layers):
            num_visible = self.dim[i]
            num_hidden = self.dim[i + 1]
            if i == num_layers - 1 and mode == 1:
                num_visible += self.output_dim
            path = os.path.join(self.path, 'rbm' + ('-%d' % i) + ('-%dh' % num_hidden) + ('-%dv' % num_visible))
            if not os.path.exists(path):
                os.mkdir(path)
            r = RBM(num_visible=num_visible, num_hidden=num_hidden, learning_rate=0.1, path=path)
            self.rbms.append(r)
        datafile = os.path.join(self.path, 'w_class')
        if os.path.isfile(datafile):
            with open(datafile, 'rb') as fp:
                self.w_class = pickle.load(fp)
            print("Load w_class Successfully!")
        else:
            # 初始化分类权重，高斯分布，均值为1，标准差为0.1
            # Initialize the w_class, using a Gaussian distribution with mean 0 and standard deviation 0.1.
            self.w_class = 0.1 * np.random.randn(self.dim[num_layers] + 1, self.output_dim)
            with open(datafile, 'wb') as fp:
                pickle.dump(self.w_class, fp)
            print("Create W_class Successfully!")
        print("Create RBM_net Successfully")

    def train_rbms(self, batch_data, batch_label=None, max_epochs_rbm=50, max_epochs_joint=200, test_set=None,
                   test_label_set=None, test_name_set=None):
        """
        Train Function
        Under mode 0, also Prediction Function
        :param batch_data:     训练集，类型为list，每个list元素为np.array，np.array是矩阵，每一行为每一个训练样本
                                training data, type: list of np.array,
                                every np.array is a matrix
                                        where each row is a training example consisting of the states of visible units.
                                i.e. every np.array is a batch of training set
        :param batch_label:    训练集标签，类型为list，每个list元素为list，该list为一个训练样本标签，与batch_data数据对应
                                training data label, type: list of list,
                                every list is a label of training example corresponding to batch_data.
        :param max_epochs_rbm:   RBM 训练的最大迭代次数，the max epochs of the RBMs training operation
        :param max_epochs_joint:   w_class 训练的最大迭代次数（此时RBM的weights也被迭代更新），mode 0下使用
                                the max epochs of the w_class training operation
                                ( weights of each RBM is updated either)
                                used under mode 0
        :param test_set:       测试集的集合, 类型为list，
                                每个list元素为list, 对该list, 其元素为np.array，np.array是矩阵，每一行为一个样本
                                the set of test data set, type: list of list of np.array,
                                every list is a test data set
                                every np.array is a matrix
                                        where each row is a example consisting of the states of visible units.
                                i.e. every np.array is a batch of visible units data set
                                used under mode 0
        :param test_label_set: 测试标签集的集合, 类型为list，
                                每个list元素为list, 对该list, 其元素为list，
                                                                    该list的元素为标签，与test_set中np.array每一行对应
                                the set of the test data label set, type: list of list of list,
                                                                        ( we call list 1 of list 2 of list 3)
                                every list2 is a test data label set
                                every list3 is the label corresponding to the row of np.array in test_set
                                used under mode 0
        :param test_name_set:  测试集名字的集合，类型为list,
                                每个list元素为字符串，是测试集名字，与test_set中各测试集的顺序对应
                                the set of the test data name, type: list of string
                                every string is name of the test data set corresponding to those in test_set
                                used under mode 0
        """
        train_data = batch_data.copy()

        for i in range(0, self.num_layers):
            # mode 1 下，最后一层（最高层）RBM的输入 visible units 为
            #                               前一层RBM输出 的 hidden units 和 标签 label 共同组成
            # In mode 1, the visible units of the top RBM consists of
            #                               the hidden units of the former RBM and the label of the test data
            if i == self.num_layers - 1 and self.mode == 1:
                train_data = list(map(lambda y: np.array(list(map(lambda x: x[0].tolist() + x[1], zip(y[0], y[1])))),
                                      zip(train_data, batch_label)))
            self.rbms[i].train(train_data, max_epochs=max_epochs_rbm)
            train_data = self.rbms[i].run_visible_for_hidden(train_data)
        print("Train RbmNet Successfully (Initial for Mode 0)")

        if self.mode == 0:
            for epoch in range(0, max_epochs_joint):
                num_batches = len(batch_data)
                counter = 0
                err_cr = 0
                for batch in range(0, num_batches):
                    data = batch_data[batch]
                    label = np.array(batch_label[batch])
                    hidden_probs = np.insert(data, 0, 1, axis=1)
                    for i in range(0, self.num_layers):
                        hidden_activations = np.dot(hidden_probs, self.rbms[i].weights)
                        hidden_probs = self._logistic(hidden_activations)
                        hidden_probs[:, 0] = 1
                    label_out = np.exp(np.dot(hidden_probs, self.w_class))
                    # label_out = np.array(list(map(lambda x: list(map(lambda y: y/np.sum(x), x)), label_out)))
                    label_out = np.divide(label_out, np.array([np.sum(label_out, axis=1).tolist()]).T)
                    counter += np.count_nonzero(np.argmax(label_out, axis=1) - np.argmax(label, axis=1))
                    # err_cr -= np.sum(np.array(list(map(lambda x: list(map(lambda y: y[0]*y[1], zip(x[0], x[1]))),
                    #                                    zip(label, np.log(label_out))))))
                    err_cr -= np.sum(np.multiply(label, np.log(label_out)))
                if self.path:
                    datafile = os.path.join(self.path, 'train_epoch.txt')
                    with open(datafile, 'at') as fp:
                        fp.write('epoch: %s, wrong num: %s, error: %s\n' % (epoch, counter, err_cr / num_batches))
                print('epoch: %s \n train: wrong: %s, error: %s' % (epoch, counter, err_cr / num_batches))

                if test_set is not None:
                    len_test_set = len(test_set)
                    test_result = [0] * len_test_set
                    test_result_err = [0] * len_test_set
                    for i in range(0, len_test_set):
                        tmp_result = self.predict(batch_test=test_set[i], batch_test_label=test_label_set[i],
                                                  test_name=test_name_set[i])
                        if epoch == 0 or tmp_result[1] < test_result[i] or (
                                        tmp_result[1] == test_result[i] and tmp_result[2] < test_result_err[i]):
                            test_result[i] = tmp_result[1]
                            test_result_err[i] = tmp_result[2]
                            datafile = os.path.join(self.path, os.path.join(test_name_set[i], 'w_class'))
                            with open(datafile, 'wb') as fp:
                                pickle.dump(self.w_class, fp)
                            for j in range(0, self.num_layers):
                                datafile = os.path.join(self.path,
                                                        os.path.join(test_name_set[i], 'weights-' + ('%d' % j)))
                                with open(datafile, 'wb') as fp:
                                    pickle.dump(self.rbms[j].weights, fp)
                            ans = tmp_result[0]
                            for j in range(0, ans.__len__()):
                                ans[j] = str(ans[j])
                            str_convert = ''.join(ans)
                            datafile = os.path.join(self.path, os.path.join(test_name_set[i], 'best_result.txt'))
                            with open(datafile, 'wt') as fp:
                                fp.write(
                                    'epoch: %d, wrong number: %d,error: %d\n' % (epoch, tmp_result[1], tmp_result[2]))
                                fp.write('%s\n' % str_convert)
                            print("Save Successfully!")
                # combine 10 batches into 1 batch for training
                tt = 0
                for batch in range(0, int(num_batches / 10)):
                    tt += 1
                    data = []
                    label = []
                    for kk in range(0, 10):
                        data += batch_data[(tt - 1) * 10 + kk].tolist()
                        label += batch_label[(tt - 1) * 10 + kk]
                    data = np.array(data)
                    # max_iter is the time of linear searches we perform conjugate gradient with
                    max_iter = 3
                    # first update top-level weights (w_class) holding other weights fixed.
                    if epoch < 6:
                        hidden_probs = np.insert(data, 0, 1, axis=1)
                        for i in range(0, self.num_layers):
                            hidden_activations = np.dot(hidden_probs, self.rbms[i].weights)
                            hidden_probs = self._logistic(hidden_activations)
                            hidden_probs[:, 0] = 1
                        vv = [self.w_class.copy()]
                        tmp = self._minimize(func=0, x=vv, parameters=[hidden_probs, label], length=max_iter)
                        self.w_class = tmp[0]
                        datafile = os.path.join(self.path, 'w_class')
                        if os.path.isfile(datafile):
                            with open(datafile, 'wb') as fp:
                                pickle.dump(self.w_class, fp)
                    else:
                        # the update all weights (w_class and weights of each RBMs)
                        vv = [0] * (self.num_layers + 1)
                        vv[0] = self.w_class.copy()
                        for i in range(0, self.num_layers):
                            vv[i + 1] = self.rbms[i].weights
                        tmp = self._minimize(func=1, x=vv, parameters=[data, label], length=max_iter)
                        self.w_class = tmp[0]
                        for i in range(0, self.num_layers):
                            self.rbms[i].weights = tmp[i + 1]
                        datafile = os.path.join(self.path, 'w_class')
                        if os.path.isfile(datafile):
                            with open(datafile, 'wb') as fp:
                                pickle.dump(self.w_class, fp)
                        for i in range(0, self.num_layers):
                            datafile = os.path.join(self.rbms[i].path, 'weights')
                            if os.path.isfile(datafile):
                                with open(datafile, 'wb') as fp:
                                    pickle.dump(self.rbms[i].weights, fp)

    def predict(self, batch_test, batch_test_label, test_name):
        """
        Prediction Function in mode 1
        :param batch_test: 可见层数据，类型为list，每个list元素为np.array，np.array是矩阵，每一行为一个样本
                            visible units data, type: list of np.array,
                            every np.array is a matrix
                                        where each row is a example consisting of the states of visible units.
                            i.e. every np.array is a batch of visible units data set
        :param batch_test_label:   测试集标签，类型为list，每个list元素为list, 该list为一个样本标签，与batch_test数据对应
                                    label, type: list of list,
                                    every list is a label of example corresponding to batch_test.
        :param test_name:  测试集名字，字符串格式
                            the name of the test set,  type: string
        :return: 一个list，第一个元素为识别类别的list，与batch_test对应
                            第二个元素为识别中错误的个数，int型
                  a list, the first element is also a list, consisting of the prediction answer,
                                                                                    corresponding to the batch_test
                          the second element is number of the wrong prediction
        """
        if self.mode == 1:
            test_data = batch_test.copy()
            for i in range(0, self.num_layers-1):
                test_data = self.rbms[i].run_visible_for_hidden(test_data)
            test_data = list(
                map(lambda y: np.array(list(map(lambda x: x + [0] * self.output_dim, y))), test_data))
            ans = self.rbms[-1].predict(test_data, soft_max=self.output_dim)
            test_num_batches = len(batch_test)
            counter = 0
            err = 0
            for batch in range(0, test_num_batches):
                counter += np.count_nonzero(np.array(ans[batch]) - np.argmax(np.array(batch_test_label[batch]), axis=1))
            if self.path:
                datafile = os.path.join(self.path, test_name)
                if not os.path.exists(datafile):
                    os.mkdir(datafile)
                datafile = os.path.join(datafile, 'test_result.txt')
                for i in range(0, ans.__len__()):
                    ans[i] = str(ans[i])
                str_convert = ''.join(ans)
                with open(datafile, 'at') as fp:
                    fp.write('%s\n' % str_convert)
            print(' %s, wrong: %s' % (test_name, counter))
            print(ans)
        else:
            test_num_batches = len(batch_test)
            counter = 0
            err_cr = 0
            ans = []
            for batch in range(0, test_num_batches):
                data = batch_test[batch]
                label = np.array(batch_test_label[batch])
                hidden_probs = np.insert(data, 0, 1, axis=1)
                for i in range(0, self.num_layers):
                    hidden_activations = np.dot(hidden_probs, self.rbms[i].weights)
                    hidden_probs = self._logistic(hidden_activations)
                    hidden_probs[:, 0] = 1
                label_out = np.exp(np.dot(hidden_probs, self.w_class))
                label_out = np.divide(label_out, np.array([np.sum(label_out, axis=1).tolist()]).T)
                predicted_ans = np.argmax(label_out, axis=1)
                counter += np.count_nonzero(predicted_ans - np.argmax(label, axis=1))
                err_cr -= np.sum(np.multiply(label, np.log(label_out)))
                ans.append(predicted_ans.tolist())
            err = err_cr / test_num_batches
            if self.path:
                datafile = os.path.join(self.path, test_name)
                if not os.path.exists(datafile):
                    os.mkdir(datafile)
                datafile = os.path.join(datafile, 'test_result.txt')
                with open(datafile, 'at') as fp:
                    fp.write('%s,%s\n' % (counter, err))
            print(' %s, wrong: %s, error: %s' % (test_name, counter, err))
            print(ans)

        return [ans, counter, err]

    @staticmethod
    def _logistic(x):
        # return 1.0 / (1 + np.exp(-x))
        return .5 * (1 + np.tanh(.5 * x))

    @staticmethod
    def _classify_init(w_class, hidden_probs, label):
        """
        the loss function of the RBM net with each RBM weights hold
        :param w_class: w_class
        :param hidden_probs: the output (hidden units) of the top RBM,
                                                suppose the input (visible units)  of RBM net is data
        :param label:  the label of data
        :return: a list, the first elements is value of the loss function with each RBM weights hold
                          the second elements is a list, consisting the partial derivative of the function
        """
        label_out = np.exp(np.dot(hidden_probs, w_class))
        # label_out = np.array(list(map(lambda x: list(map(lambda y: y/np.sum(x), x)), label_out)))
        label_out = np.divide(label_out, np.array([np.sum(label_out, axis=1).tolist()]).T)
        # f = - np.sum(np.array(list(map(lambda x: list(map(lambda y: y[0]*y[1], zip(x[0], x[1]))),
        #                                zip(label, np.log(label_out))))))
        f = - np.sum(np.multiply(label, np.log(label_out)))
        df = np.dot(hidden_probs.T, label_out - label)
        return [f, [df]]

    def _classify(self, w_class, weights, data, label):
        """
        the loss function of the RBM net
        :param w_class: w_class
        :param weights: a list, consisting of weights of each RBM
        :param data: the input (visible units) of the first RBM
        :param label: the label of the data
        :return: a list, the first elements is value of the loss function
                          the second elements is a list, consisting the partial derivative of the function
                                                                            corresponding to w_class and weights[i]
        """
        # hidden_probs is a list, the i-th elements is the input of the i-th RBM or the output of the i-1th RBM
        hidden_probs = [np.insert(data, 0, 1, axis=1)] * (self.num_layers + 1)  # 0 data 1
        for i in range(0, self.num_layers):
            hidden_activations = np.dot(hidden_probs[i], weights[i])
            hidden_probs[i + 1] = self._logistic(hidden_activations)
            hidden_probs[i + 1][:, 0] = 1
        label_out = np.exp(np.dot(hidden_probs[self.num_layers], w_class))
        # label_out = np.array(list(map(lambda x: list(map(lambda y: y/np.sum(x), x)), label_out)))
        # f = - np.sum(np.array(list(map(lambda x: list(map(lambda y: y[0]*y[1], zip(x[0], x[1]))),
        #                                zip(label, np.log(label_out))))))
        label_out = np.divide(label_out, np.array([np.sum(label_out, axis=1).tolist()]).T)
        f = - np.sum(np.multiply(label, np.log(label_out)))
        io = label_out - label
        dw_class = np.dot(hidden_probs[self.num_layers].T, io)
        tmp1 = np.dot(io, w_class.T)
        # tmp2 = np.array(list(map(lambda x: list(map(lambda y: 1-y, x)), hidden_probs[self.num_layers])))
        tmp2 = np.subtract(1, hidden_probs[self.num_layers])
        # Ix = np.array(list(map(lambda x: list(map(lambda y: y[0]*y[1]*y[2], zip(x[0], x[1], x[2]))),
        #                        zip(tmp1, hidden_probs[self.num_layers], tmp2))))
        ix = np.multiply(np.multiply(tmp1, hidden_probs[self.num_layers]), tmp2)
        dw = [0] * (self.num_layers + 1)
        dw[0] = dw_class
        for i in range(0, self.num_layers):
            dw[self.num_layers - i] = np.dot(hidden_probs[self.num_layers - 1 - i].T, ix)
            if i < self.num_layers - 1:
                tmp1 = np.dot(ix, weights[self.num_layers - 1 - i].T)
                # tmp2 = np.array(list(map(lambda x: list(map(lambda y: 1-y, x)), hidden_probs[self.num_layers-1-i])))
                tmp2 = np.subtract(1, hidden_probs[self.num_layers - 1 - i])
                # Ix = list(map(lambda x: list(map(lambda y: y[0]*y[1]*y[2], zip(x[0], x[1], x[2]))),
                #               zip(tmp1, hidden_probs[self.num_layers-1-i], tmp2)))
                ix = np.multiply(np.multiply(tmp1, hidden_probs[self.num_layers - 1 - i]), tmp2)
        return [f, dw]

    def _minimize(self, func, x, parameters, length):
        """
         Minimize a differentiable multivariate function
        :param func: the type of function, 1 means _classify, 0 means _classify_init
        :param x: the initial value of the variation, here, it is a list,
                        if func = 0, then its element is w_class
                        if func = 1, then its elements are w_class, weights of each RBM
        :param parameters: the unchanged parameters of function represented by func
        :param length: the maximum number of line searches
        :return: the result with which the function value is smaller than before
                    here, it is a list
                    if func = 0, then its element is w_class
                    if func = 1, then its elements are w_class, weights of each RBM
        """

        i = 0
        is_failed = 0
        if func:
            tmp = self._classify(w_class=x[0], weights=x[1:], data=parameters[0], label=parameters[1])
        else:
            tmp = self._classify_init(w_class=x[0], hidden_probs=parameters[0], label=parameters[1])
        f0 = tmp[0]
        df0 = tmp[1]
        s = list(map(lambda x: - x, df0))
        d0 = - sum(list(map(lambda x: np.sum(np.multiply(x, x)), s)))
        x3 = 1.0 / (1 - d0)
        while i < length:
            i += 1
            x0 = x.copy()
            F0 = f0
            dF0 = df0.copy()
            M = MAX
            while 1:
                x2 = 0
                f2 = f0
                d2 = d0
                f3 = f0
                df3 = df0.copy()
                success = 0
                while (not success) and M > 0:
                    M -= 1
                    i += 1
                    newx = list(map(lambda x: x[0] + x3 * x[1], zip(x, s)))
                    if func:
                        tmp = self._classify(w_class=newx[0], weights=newx[1:], data=parameters[0], label=parameters[1])
                    else:
                        tmp = self._classify_init(w_class=newx[0], hidden_probs=parameters[0], label=parameters[1])
                    f3 = tmp[0]
                    df3 = tmp[1]
                    errf = np.zeros_like(f3)
                    errf = np.count_nonzero(np.isinf(f3, errf)) > 0
                    if errf:
                        x3 = (x2 + x3) / 2.0
                        continue
                    errf = np.zeros_like(f3)
                    errf = np.count_nonzero(np.isnan(f3, errf)) > 0
                    if errf:
                        x3 = (x2 + x3) / 2.0
                        continue
                    for element in df3:
                        errdf = np.zeros_like(element)
                        errdf = np.count_nonzero(np.isinf(element, errdf)) > 0
                        if errdf:
                            x3 = (x2 + x3) / 2.0
                            break
                        errdf = np.zeros_like(element)
                        errdf = np.count_nonzero(np.isnan(element, errdf)) > 0
                        if errdf:
                            x3 = (x2 + x3) / 2.0
                            break
                    if errdf:
                        continue
                    success = 1
                if f3 < F0:
                    x0 = list(map(lambda x: x[0] + x3 * x[1], zip(x, s)))
                    F0 = f3
                    dF0 = df3.copy()
                d3 = sum(list(map(lambda x: np.sum(np.multiply(x[0], x[1])), zip(df3, s))))
                if d3 > SIG * d0 or f3 > (f0 + x3 * RHO * d0) or M == 0:
                    break
                x1 = x2
                f1 = f2
                d1 = d2
                x2 = x3
                f2 = f3
                d2 = d3
                A = 6 * (f1 - f2) + 3 * (d2 + d1) * (x2 - x1)
                B = 3 * (f2 - f1) - (2 * d1 + d2) * (x2 - x1)
                x3 = x1 - d1 * ((x2 - x1) ** 2.0) / (B + ((B * B - A * d1 * (x2 - x1)) ** (1 / 2.0)))
                if (not np.isreal(x3)) or (np.isnan(x3)) or (np.isinf(x3)) or x3 < 0:
                    x3 = x2 * EXT
                else:
                    if x3 > x2 * EXT:
                        x3 = x2 * EXT
                    else:
                        if x3 < (x2 + INT * (x2 - x1)):
                            x3 = (x2 + INT * (x2 - x1))

            while (abs(d3) > -SIG * d0 or f3 > f0 + x3 * RHO * d0) and M > 0:
                if d3 > 0 or f3 > (f0 + x3 * RHO * d0):
                    x4 = x3
                    f4 = f3
                    d4 = d3
                else:
                    x2 = x3
                    f2 = x3
                    d2 = d3
                if f4 > f0:
                    x3 = x2 - (0.5 * d2 * ((x4 - x2) ** 2)) / (f4 - f2 - d2 * (x4 - x2))
                else:
                    A = 6 * (f2 - f4) / (x4 - x2) + 3 * (d4 + d2)
                    B = 3 * (f4 - f2) - (2 * d2 + d4) * (x4 - x2)
                    x3 = x2 + ((B * B - A * d2 * ((x4 - x2) ** 2)) ** (1 / 2.0) - B) / A
                if np.isnan(x3) or np.isinf(x3):
                    x3 = (x2 + x4) / 2.0
                x3 = max(min(x3, (x4 - INT * (x4 - x2))), (x2 + INT * (x4 - x2)))
                newx = list(map(lambda x: x[0] + x3 * x[1], zip(x, s)))
                if func:
                    tmp = self._classify(w_class=newx[0], weights=newx[1:], data=parameters[0], label=parameters[1])
                else:
                    tmp = self._classify_init(w_class=newx[0], hidden_probs=parameters[0], label=parameters[1])
                f3 = tmp[0]
                df3 = tmp[1]
                if f3 < F0:
                    x0 = list(map(lambda x: x[0] + x3 * x[1], zip(x, s)))
                    F0 = f3
                    dF0 = df3.copy()
                M = M - 1
                d3 = sum(list(map(lambda x: np.sum(np.multiply(x[0], x[1])), zip(df3, s))))

            if (abs(d3) < -SIG * d0) and (f3 < f0 + x3 * RHO * d0):
                x = list(map(lambda x: x[0] + x3 * x[1], zip(x, s)))
                f0 = f3
                u33 = sum(list(map(lambda x: np.sum(np.multiply(x[0], x[1])), zip(df3, df3))))
                u03 = sum(list(map(lambda x: np.sum(np.multiply(x[0], x[1])), zip(df0, df3))))
                u00 = sum(list(map(lambda x: np.sum(np.multiply(x[0], x[1])), zip(df0, df0))))
                s = list(map(lambda x: (u33 - u03) / u00 * x[0] - x[1], zip(s, df3)))
                df0 = df3.copy()
                d3 = d0
                d0 = sum(list(map(lambda x: np.sum(np.multiply(x[0], x[1])), zip(df0, s))))
                if d0 > 0:
                    s = list(map(lambda x: - x, df0))
                    d0 = -sum(list(map(lambda x: np.sum(np.multiply(x, x)), s)))
                realmin = np.finfo(np.double).tiny
                x3 = x3 * min(RATIO, (d3 / (d0 - realmin)))
                is_failed = 0
            else:
                x = x0.copy()
                f0 = F0
                df0 = dF0.copy()
                if is_failed or i > length:
                    break
                s = list(map(lambda x: - x, df0))
                d0 = -sum(list(map(lambda x: np.sum(np.multiply(x, x)), s)))
                x3 = 1 / (1 - d0)
                is_failed = 1
        return x
