# -*-coding:utf-8-*-
from __future__ import print_function

import numpy as np
import pickle
import os


# Single Layer Restricted Boltzmann Machine
class RBM:
    def __init__(self, num_visible, num_hidden, learning_rate=0.1, path=None):
        """
        初始化函数
        Initial Function
        :param num_visible:  可见层单元个数，the number of visible units
        :param num_hidden:  隐含层单元个数，the number of hidden units
        :param learning_rate:  学习率，the learning rate of RBM
        :param path:   该RBM参数存储的路径
                        the path where we store the parameters of RBM

        RBM类成员：
            weights：   weights 是 RBM 的网络权重矩阵，其大小为 ( 1 + num_visible ) * ( 1 + num_hidden)
                        第一行为隐含层偏置权重，第一列为可见层偏置权重，第一行第一列永远为0
                        第二行第二列起至矩阵尾为可见层与隐含层之间的边的权重
                        weights is the matrix of size ( 1 + num_visible ) * ( 1 + num_hidden)
                        the first row of "weights" is the hidden bias, the first column of "weights" is the visible bias
                        the rest part of "weights" is the weight matrix of edges between visible units and hidden units
            weightsinc: weightsinc 是训练过程中weights矩阵的增量
                        weightsinc is the increase (or change) of "weights" in every epoch of training
        """

        self.num_hidden = num_hidden
        self.num_visible = num_visible
        self.learning_rate = learning_rate
        self.path = path

        # 查看是否有存档，如果有，则载入参数weights和weightsinc
        # Check whether the parameter file exists; if so, load the data
        datafile = os.path.join(self.path, 'weights')
        if os.path.isfile(datafile):
            with open(datafile, 'rb') as fp:
                self.weights = pickle.load(fp)
            print("Load Weights Successfully!")
            datafile = os.path.join(self.path, 'weightsinc')
            with open(datafile, 'rb') as fp:
                self.weightinc = pickle.load(fp)
            print("Load WeightInc Successfully!")
        else:
            # 初始化权重 "weights"，高斯分布，均值为1，标准差为0.1
            # Initialize the weights, using a Gaussian distribution with mean 0 and standard deviation 0.1.
            self.weights = 0.1 * np.random.randn(self.num_visible, self.num_hidden)
            # Insert "weights" for the bias units into the first row and first column 插入偏置单元
            self.weights = np.insert(self.weights, 0, 0, axis=0)
            self.weights = np.insert(self.weights, 0, 0, axis=1)
            with open(datafile, 'wb') as fp:
                pickle.dump(self.weights, fp)
            print("Create Weights Successfully!")
            # 初始化 "weightsinc"，0矩阵
            # Initialize the weightsinc with zero matrix
            self.weightinc = np.zeros([self.num_visible + 1, self.num_hidden + 1])
            datafile = os.path.join(self.path, 'weightsinc')
            with open(datafile, 'wb') as fp:
                pickle.dump(self.weightinc, fp)
            print("Create WeightInc Successfully!")

    def train(self, batch_data, max_epochs=50):
        """
        Train the RBM
        :param batch_data: 训练集，类型为list，每个list元素为np.array，np.array是矩阵，每一行为每一个训练样本
                            training data, type: list of np.array,
                            every np.array is a matrix
                                        where each row is a training example consisting of the states of visible units.
                            i.e. every np.array is a batch of training set
        :param max_epochs: 训练最大迭代次数, the max epochs of the training operation
        """
        # Initialization
        # weightcost times weightsinc and then be added to the normal gradient (i.e. weightsinc)
        # weightcost 乘以增量weightsinc 加上新的weightinc, 是为了weight-decay
        # weightcost ranges from 0.01 to 0.00001
        weightcost = 0.0002
        # Momentum is a simple method for increasing the speed of learning when the objective function
        # contains long, narrow and fairly straight ravines with a gentle but consistent gradient along the floor
        # of the ravine and much steeper gradients up the sides of the ravine.
        initialmomentum = 0.5
        finalmomentum = 0.9
        count = 0
        for epoch in range(0, max_epochs):
            errorsum = 0
            for data in batch_data:
                num_examples = data.shape[0]
                # 第一列加入偏置单元 1
                # Insert bias units of 1 into the first column.
                data = np.insert(data, 0, 1, axis=1)
                # Gibbs Sample
                # 从 data 中 采样 sample 得到 隐单元 hidden units
                # (This is the "positive CD phase", aka 正相.)
                pos_hidden_activations = np.dot(data, self.weights)
                pos_hidden_probs = self._logistic(pos_hidden_activations)
                # Fix the bias unit 修正偏置单元
                pos_hidden_probs[:, 0] = 1
                pos_hidden_states = pos_hidden_probs > np.random.rand(num_examples, self.num_hidden + 1)
                pos_associations = np.dot(data.T, pos_hidden_probs)
                # 从隐单元 hidden units 采样 sample，重构 reconstruct 显单元 visible units
                # (This is the "negative CD phase", aka 负相.)
                neg_visible_activations = np.dot(pos_hidden_states, self.weights.T)
                neg_visible_probs = self._logistic(neg_visible_activations)
                # Fix the bias unit 修正偏置单元
                neg_visible_probs[:, 0] = 1
                neg_hidden_activations = np.dot(neg_visible_probs, self.weights)
                neg_hidden_probs = self._logistic(neg_hidden_activations)
                # Fix the bias unit 修正偏置单元
                neg_hidden_probs[:, 0] = 1
                neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)
                error = np.sum((data - neg_visible_probs) ** 2)
                errorsum = error + errorsum
                # 选择 momentum
                if epoch > 5:
                    momentum = finalmomentum
                else:
                    momentum = initialmomentum
                # Update weights 更新权重
                delta = (pos_associations - neg_associations) / num_examples
                vishid = self.weights[1:self.num_visible + 1, 1:self.num_hidden + 1]
                vishid = np.insert(vishid, 0, 0, axis=0)
                vishid = np.insert(vishid, 0, 0, axis=1)
                self.weightinc = momentum * self.weightinc + self.learning_rate * (delta - weightcost * vishid)
                # 确保无关项为 0
                self.weightinc[0, 0] = 0
                self.weights += self.weightinc
                self.weights[0, 0] = 0
                count += 1
                print("Count %s: error is %s" % (count, error))
                # Save weights and error 保存权值和误差
                if self.path:
                    datafile = os.path.join(self.path, 'weights')
                    with open(datafile, 'wb') as fp:
                        pickle.dump(self.weights, fp)
                    datafile = os.path.join(self.path, 'count.txt')
                    with open(datafile, 'at') as fp:
                        fp.write("%s,%s\n" % (count, error))
            if self.path:
                datafile = os.path.join(self.path, 'epoch.txt')
                with open(datafile, 'at') as fp:
                    fp.write("%s,%s\n" % (epoch, errorsum))

    def run_visible_for_hidden(self, batch_data):
        """
        Assuming the RBM has been trained (so that weights for the network have been learned),
        run the network on a set of visible units, to get probabilities of the hidden units.
        :param batch_data: 可见层数据，类型为list，每个list元素为np.array，np.array是矩阵，每一行为一个样本
                            visible units data, type: list of np.array,
                            every np.array is a matrix
                                        where each row is a example consisting of the states of visible units.
                            i.e. every np.array is a batch of visible units data set
        :return:   隐含单元的logistic概率，类型为list，每个list元素为np.array，与输入数据相对应
                    the probabilities of the hidden units, type: list of np.array,
                    every np.array is a batch of hidden units data set, corresponding to the input
        """
        batch_pos_hidden_probs = []
        for data in batch_data:
            # 第一列加入偏置单元 1
            # Insert bias units of 1 into the first column of data.
            data = np.insert(data, 0, 1, axis=1)
            # Calculate the activations of the hidden units.
            hidden_activations = np.dot(data, self.weights)
            # Calculate the probabilities of turning the hidden units on.
            hidden_probs = self._logistic(hidden_activations)
            pos_hidden_probs = hidden_probs[:, 1:]
            batch_pos_hidden_probs.append(pos_hidden_probs)
        return batch_pos_hidden_probs

    def run_hidden_for_visible(self, batch_data):
        """
        Assuming the RBM has been trained (so that weights for the network have been learned),
        run the network on a set of hidden units, to get probabilities of the visible units.
        :param batch_data: 隐含层数据，类型为list，每个list元素为np.array，np.array是矩阵，每一行为一个样本
                            hidden units data, type: list of np.array,
                            every np.array is a matrix
                                        where each row is a example consisting of the states of hidden units.
                            i.e. every np.array is a batch of hidden units data set
        :return:   可见单元的logistic概率，类型为list，每个list元素为np.array，与输入数据相对应
                    the probabilities of the visible units, type: list of np.array,
                    every np.array is a batch of visible units data set, corresponding to the input
        """
        batch_neg_visible_probs = []
        for data in batch_data:
            # Insert bias units of 1 into the first column of data.
            data = np.insert(data, 0, 1, axis=1)
            # Calculate the activations of the visible units.
            visible_activations = np.dot(data, self.weights.T)
            # Calculate the probabilities of turning the visible units on.
            visible_probs = self._logistic(visible_activations)
            neg_visible_probs = visible_probs[:, 1:]
            batch_neg_visible_probs.append(neg_visible_probs)
        return batch_neg_visible_probs

    def predict(self, batch_data, soft_max=10):
        """
        Assuming the RBM has been trained (so that weights for the network have been learned),
        run the network on a set of test data, to get recognition results (only perform digits recognition)
        This prediction method is especially designed for the visible units including the label(softmax)
        :param batch_data: 可见层数据，类型为list，每个list元素为np.array，np.array是矩阵，每一行为一个样本
                            visible units data, type: list of np.array,
                            every np.array is a matrix
                                        where each row is a example consisting of the states of visible units.
                            i.e. every np.array is a batch of visible units data set
        :param soft_max:   标签的维度, 为4或10，4维是0-9的二进制表示，10维的每一维（只能为0或1）对应是否属于相应数字类别
                            the dimension of label, only can take value of 4 or 10
                            4 means the label is expressed as binary
                            10 means the state of each dimension infer whether it belongs to that class
        :return:   分类结果，类型为list，其元素为list, 该list元素为标签，与输入数据相对应
                    the classification result, type: list of list of int,
                    list2 is a batch of answers, corresponding to the input
        """
        final_ans = []
        for data in batch_data:
            ans = []
            num_examples = data.shape[0]
            data = np.insert(data, 0, 1, axis=1)
            data = np.split(data, num_examples)
            for item in data:
                hidden_activations = np.dot(item, self.weights)
                vbias_energy = hidden_activations[0, 0]
                hidden_probs = self._logfree(hidden_activations)
                hidden_probs[:, 0] = 0
                free_energy = - np.sum(hidden_probs) - vbias_energy
                min_free_energy = free_energy
                tmp_ans = 0
                for number in range(1, 10):
                    tmpitem = item.copy()
                    if soft_max == 10:
                        tmpitem[0, self.num_visible - 9:self.num_visible + 1] = 0
                        tmpitem[0, self.num_visible - (9 - number)] = 1
                    else:
                        if soft_max == 4:
                            label = bin(number)
                            label = label[::-1]
                            length = len(label)
                            for i in range(0, length - 3 + 1):
                                tmpitem[0, self.num_visible + i - 3] = int(label[i])
                            if length != 6:
                                for i in range(1, 6 - length + 1):
                                    tmpitem[0, self.num_visible - (6 - length) + i] = 0
                    hidden_activations = np.dot(tmpitem, self.weights)
                    vbias_energy = hidden_activations[0, 0]
                    hidden_probs = self._logfree(hidden_activations)
                    hidden_probs[:, 0] = 0
                    free_energy = - np.sum(hidden_probs) - vbias_energy
                    if free_energy < min_free_energy:
                        tmp_ans = number
                        min_free_energy = free_energy
                ans.append(tmp_ans)
            final_ans.append(ans)
        return final_ans

    @staticmethod
    def _logistic(x):
        # np.tanh is more stable than np.exp in numpy
        # return 1.0 / (1 + np.exp(-x))
        return .5 * (1 + np.tanh(.5 * x))

    @staticmethod
    def _logfree(x):
        return np.log(1 + np.exp(x))
