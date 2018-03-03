# -*-coding:utf-8-*-
from __future__ import print_function

import numpy as np
import os

from .rbm import RBM


# 多个RBM组合类
class RbmForest:
    def __init__(self, num_visible, num_hidden, num_output=10, learning_rate=0.1, path=None):
        """
        Because we only recognize 10 numbers, so the RBM_each consists of 10 RBMs
        :param num_visible: 可见层单元个数，the number of visible units
        :param num_hidden: 隐含层单元个数，the number of hidden units
        :param num_output: 输出标签维度， the number of output labels
        :param learning_rate: 学习率，the learning rate of RBM
        :param path:   所有RBM参数存储的路径
                        the path where we store the parameters of RBM
        """
        self.num_hidden = num_hidden
        self.num_visible = num_visible
        self.num_output = num_output
        self.learning_rate = learning_rate
        self.path = path
        self.rbms = []
        for i in range(0, self.num_output):
            path = os.path.join(self.path, ('rbm-%d' % i))
            os.mkdir(path)
            r = RBM(num_visible=num_visible, num_hidden=num_hidden, learning_rate=learning_rate, path=path)
            self.rbms.append(r)

    def train(self, train_data, batch_size=100, max_epochs=50):
        """
        训练函数 Train Function
        :param train_data:  训练集，类型为list，有10个元素，对应10个数字，
                             元素为np.array, np.array是矩阵，每一行为每一个训练样本
                             training data, type: list of np.array,
                             every np.array is a matrix
                                        where each row is a training example consisting of the states of visible units.
                             i.e. each np.array is a training set of a class
        :param batch_size:  每个训练集要分成batches进行训练，每个batches含有的样本数为batch_size
                         the number of training example in one batch of a training set of a class
        :param max_epochs: 训练最大迭代次数, the max epochs of the training operation
        """
        for i in range(0, self.num_output):
            batch_data = np.array_split(train_data[i], train_data[i].shape[0] / batch_size)
            r = self.rbms[i]
            r.train(batch_data, max_epochs=max_epochs)
            print(r.weights)
            print("Train RBM %d Successfully" % i)

    def predict(self, test):
        """
        Assuming the RBM has been trained (so that weights for the network have been learned),
        run the network on a set of test data, to get recognition results (only perform digits recognition)
        :param test: 测试集，类型为list，元素为np.array，np.array矩阵只有1行，为一个样本
                      visible units data, type: list of np.array,
                      each np.array consists of one row and is a example consisting of the states of visible units.
        :return: the prediction result, type:list
        """
        ans = []
        for item in test:
            minerror = 0
            tmpans = 0
            tmpitem = item.copy()
            tmpitem = [tmpitem]
            for number in range(0, self.num_output):
                r = self.rbms[number]
                hidden_probs = r.run_visible_for_hidden(tmpitem)
                visible_probs_batches = r.run_hidden_for_visible(hidden_probs)
                visible_probs = visible_probs_batches[0]
                error = np.sum(np.square(item - visible_probs))
                if number == 0:
                    minerror = error
                    tmpans = 0
                else:
                    if error < minerror:
                        tmpans = number
                        minerror = error
            ans.append(tmpans)
        return ans