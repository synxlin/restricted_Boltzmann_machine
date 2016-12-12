from __future__ import print_function
import numpy as np
import pickle
from mnist import MNIST
from PIL import Image

# RBM 类
class RBM:
    def __init__(self, num_visible, num_hidden, learning_rate=0.1, Path=None):
        """
        初始化函数
        Initial Function
        :param num_visible:  可见层单元个数，the number of visible units
        :param num_hidden:  隐含层单元个数，the number of hidden units
        :param learning_rate:  学习率，the learning rate of RBM
        :param Path:   该RBM参数存储的路径，以//结尾的字符串
                        the path where we store the parameters of RBM, and it ends with //

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
        self.path = Path

        # 查看是否有存档，如果有，则载入参数weights和weightsinc
        # Check whether the parameter file exists; if so, load the data
        import os
        datafile = self.path + 'weights'
        if os.path.isfile(datafile):
            with open(datafile, 'rb') as fp:
                self.weights = pickle.load(fp)
            print("Load Weights Successfully!")
            datafile = self.path + 'weightsinc'
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
            self.weightinc = np.zeros([self.num_visible+1, self.num_hidden+1])
            datafile = self.path + 'weightsinc'
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
                vishid = self.weights[1:self.num_visible+1, 1:self.num_hidden+1]
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
                    datafile = self.path+'weights'
                    with open(datafile, 'wb') as fp:
                        pickle.dump(self.weights, fp)
                    datafile = self.path+'count.txt'
                    with open(datafile, 'at') as fp:
                        fp.write("%s,%s\n" % (count, error))
            if self.path:
                datafile = self.path+'epoch.txt'
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
                        tmpitem[0, self.num_visible-9:self.num_visible+1] = 0
                        tmpitem[0, self.num_visible-(9-number)] = 1
                    else:
                        if soft_max == 4:
                            label = bin(number)
                            label = label[::-1]
                            length = len(label)
                            for i in range(0, length - 3 + 1):
                                tmpitem[0, self.num_visible+i-3] = int(label[i])
                            if length != 6:
                                for i in range(1, 6 - length + 1):
                                    tmpitem[0, self.num_visible-(6-length)+i] = 0
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

    def _logistic(self, x):
        # np.tanh is more stable than np.exp in numpy
        # return 1.0 / (1 + np.exp(-x))
        return .5 * (1 + np.tanh(.5*x))

    def _logfree(self, x):
        return np.log(1 + np.exp(x))

# 多层RBM网络类
class RBM_net:
    def __init__(self, layers=3, dim=None, learning_rate=0.1, Path=None, mode=0):
        """
        初始化函数
        Initial Function
        :param layers:  RBM的层数（个数），RBM级联，即上一个RBM的 hidden units 为下个RBM的 visible units
                         the layers or numbers of RBM
                         the hidden units of the former RBMis the visible units of the latter RBM
        :param dim:     每个RBM的可见层和隐含层单元数，类型为list
                         第i个元素为第i个RBM可见单元个数，第i+1个元素为第i个RBM隐含单元个数
                         the visible units number and hidden units number of each RBM, type: list
                         the i-th elements of list is the the visible units number of the i-th RBM
                         the i+1-th elements of list is the the hidden units number of the i-th RBM
        :param learning_rate: RBM的学习率, learning rate of RBM
        :param Path:    该RBM网络的参数存储路径，以//结尾的字符串
                         the path where we store the parameters of RBM net, and it ends with //
        :param mode:    该RBM net的模式，0 代表标签不作为可见单元  1 代表 标签作为可见单元
                         label is used as visible units under mode 1; otherwise, mode 0
        RBM_net 类成员
        w_class:    mode 0 下，从最后一层隐单元转移至标签输出的权重矩阵
                    （1 + num_visible of the top RBM) * the dimension of label(softmax)
                    the weight matrix between the hidden units of the top RBM and the label(softmax)
        """
        self.layers = layers
        if not dim:
            self.dim = [784, 500, 500, 2000, 10]
        else:
            self.dim = dim
        self.learning_rate = learning_rate
        self.Path = Path
        self.rbms = []
        self.mode = mode
        for i in range(0, layers):
            num_visible = dim[i]
            num_hidden = dim[i+1]
            if i == layers - 1 and mode == 1:
                num_visible += dim[layers+1]
            path = self.Path + 'rbm' + ('-%d' % i) + ('-%dh' % num_hidden) + ('-%dv' % num_visible)
            import os
            if not os.path.exists(path):
                os.mkdir(path)
            path += '\\'
            r = RBM(num_visible=num_visible, num_hidden=num_hidden, learning_rate=0.1, Path=path)
            self.rbms.append(r)
        datafile = self.Path + 'w_class'
        import os
        if os.path.isfile(datafile):
            with open(datafile, 'rb') as fp:
                self.w_class = pickle.load(fp)
            print("Load w_class Successfully!")
        else:
            # 初始化分类权重，高斯分布，均值为1，标准差为0.1
            # Initialize the w_class, using a Gaussian distribution with mean 0 and standard deviation 0.1.
            self.w_class = 0.1 * np.random.randn(dim[layers]+1, dim[layers+1])
            with open(datafile, 'wb') as fp:
                pickle.dump(self.w_class, fp)
            print("Create W_class Successfully!")
        print("Create RBM_net Successfully")

    def train_rbms(self, batch_data, batch_label=None, max_epochs_1=50, max_epochs_2=200, test_set=None, test_label_set=None, test_name_set=None):
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
        :param max_epochs_1:   RBM 训练的最大迭代次数，the max epochs of the RBMs training operation
        :param max_epochs_2:   w_class 训练的最大迭代次数（此时RBM的weights也被迭代更新），mode 0下使用
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

        for i in range(0, self.layers):
            # mode 1 下，最后一层（最高层）RBM的输入 visible units 为
            #                               前一层RBM输出 的 hidden units 和 标签 label 共同组成
            # In mode 1, the visible units of the top RBM consists of
            #                               the hidden units of the former RBM and the label of the test data
            if i == self.layers - 1 and self.mode == 1:
                train_data = list(map(lambda y: np.array(list(map(lambda x: x[0].tolist()+x[1], zip(y[0], y[1])))), zip(train_data, batch_label)))
            self.rbms[i].train(train_data, max_epochs=max_epochs_1)
            train_data = self.rbms[i].run_visible_for_hidden(train_data)
        print("Train RBM_net Successfully")

        if self.mode == 0:
            if not (test_set == None):
                    num_teset_set = len(test_set)
                    test_result = [0] * num_teset_set
                    test_result_err = [0] * num_teset_set
            for epoch in range(0, max_epochs_2):
                num_batches = len(batch_data)
                counter = 0
                err_cr = 0
                for batch in range(0, num_batches):
                    data = batch_data[batch]
                    label = np.array(batch_label[batch])
                    hidden_probs = np.insert(data, 0, 1, axis=1)
                    for i in range(0, self.layers):
                        hidden_activations = np.dot(hidden_probs, self.rbms[i].weights)
                        hidden_probs = self._logistic(hidden_activations)
                        hidden_probs[:, 0] = 1
                    label_out = np.exp(np.dot(hidden_probs, self.w_class))
                    # label_out = np.array(list(map(lambda x: list(map(lambda y: y/np.sum(x), x)), label_out)))
                    label_out = np.divide(label_out, np.array([np.sum(label_out, axis=1).tolist()]).T)
                    J = np.argmax(label_out, axis=1)
                    J1 = np.argmax(label, axis=1)
                    counter += np.count_nonzero(J-J1)
                    # err_cr -= np.sum(np.array(list(map(lambda x: list(map(lambda y: y[0]*y[1], zip(x[0], x[1]))), zip(label, np.log(label_out))))))
                    err_cr -= np.sum(np.multiply(label, np.log(label_out)))
                if self.Path:
                    datafile = self.Path+'train_epoch.txt'
                    with open(datafile, 'at') as fp:
                        fp.write('epoch: %s, wrong num: %s, error: %s\n' % (epoch, counter, err_cr/num_batches))
                print('epoch: %s \n train: wrong: %s, error: %s' % (epoch, counter, err_cr/num_batches))

                if not (test_set == None):
                    num_teset_set = len(test_set)
                    for i in range(0, num_teset_set):
                        tmp_result = r.predict(batch_test=test_set[i], batch_test_label=test_label_set[i], test_name=test_name_set[i])
                        if epoch == 0 or tmp_result[1] < test_result[i] or (tmp_result[1] == test_result[i] and tmp_result[2] < test_result_err[i]):
                            test_result[i] = tmp_result[1]
                            test_result_err[i] = tmp_result[2]
                            datafile = self.Path + test_name_set[i] + '\\w_class'
                            with open(datafile, 'wb') as fp:
                                pickle.dump(self.w_class, fp)
                            for j in range(0, self.layers):
                                datafile = self.Path + test_name_set[i] + '\\weights-' + ('%d' % j)
                                with open(datafile, 'wb') as fp:
                                    pickle.dump(self.rbms[j].weights, fp)
                            ans = tmp_result[0]
                            for j in range(0, ans.__len__()):
                                ans[j] = str(ans[j])
                            str_convert = ''.join(ans)
                            datafile = self.Path + test_name_set[i] + '\\best_result.txt'
                            with open(datafile, 'wt') as fp:
                                fp.write('epoch: %d, wrong number: %d,error: %d\n' % (epoch, tmp_result[1], tmp_result[2]))
                                fp.write('%s\n' % str_convert)
                            print('Save Successfully!')
                # combine 10 batches into 1 batch for training
                tt = 0
                for batch in range(0, int(num_batches/10)):
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
                        for i in range(0, self.layers):
                            hidden_activations = np.dot(hidden_probs, self.rbms[i].weights)
                            hidden_probs = self._logistic(hidden_activations)
                            hidden_probs[:, 0] = 1
                        VV = [self.w_class.copy()]
                        tmp = self._minimize(func=0, x=VV, parameters=[hidden_probs, label], length=max_iter)
                        self.w_class = tmp[0]
                        import os
                        datafile = self.Path + 'w_class'
                        if os.path.isfile(datafile):
                            with open(datafile, 'wb') as fp:
                                pickle.dump(self.w_class, fp)
                    else:
                        # the update all weights (w_class and weights of each RBMs)
                        VV = [0] * (self.layers + 1)
                        VV[0] = self.w_class.copy()
                        for i in range(0, self.layers):
                            VV[i+1] = self.rbms[i].weights
                        tmp = self._minimize(func=1, x=VV, parameters=[data, label], length=max_iter)
                        self.w_class = tmp[0]
                        for i in range(0, self.layers):
                            self.rbms[i].weights = tmp[i+1]
                        import os
                        datafile = self.Path + 'w_class'
                        if os.path.isfile(datafile):
                            with open(datafile, 'wb') as fp:
                                pickle.dump(self.w_class, fp)
                        for i in range(0, self.layers):
                            datafile = self.rbms[i].path + 'weights'
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
            for i in range(0, self.layers):
                if i == self.layers - 1:
                    test_data = list(map(lambda y: np.array(list(map(lambda x: x+[0]*self.dim[self.layers+1], y.tolist()))), test_data))
                    ans = self.rbms[i].predict(test_data, soft_max=self.dim[self.layers+1])
                else:
                    test_data = self.rbms[i].run_visible_for_hidden(test_data)
            test_num_batches = len(batch_test)
            counter = 0
            err = 0
            for batch in range(0, test_num_batches):
                J = np.array(ans[batch])
                if self.dim[self.layers+1] == 10:
                    J1 = np.array(batch_test_label[batch])
                    J1 = np.argmax(J1, axis=1)
                else:
                    J1 = np.array(list(map(lambda x: x[0] + 2 * x[1] + 4 * x[2] + 8 * x[3], batch_test_label[batch])))
                counter += np.count_nonzero(J-J1)
            if self.Path:
                datafile = self.Path + 'log.txt'
                for i in range(0, ans.__len__()):
                    ans[i] = str(ans[i])
                str_convert = ''.join(ans)
                with open(datafile, 'at') as fp:
                    fp.write('\n %s, wrong number: %s\n' % (test_name, counter))
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
                for i in range(0, self.layers):
                    hidden_activations = np.dot(hidden_probs, self.rbms[i].weights)
                    hidden_probs = self._logistic(hidden_activations)
                    hidden_probs[:, 0] = 1
                label_out = np.exp(np.dot(hidden_probs, self.w_class))
                #label_out = np.array(list(map(lambda x: list(map(lambda y: y/np.sum(x), x)), label_out)))
                label_out = np.divide(label_out, np.array([np.sum(label_out,axis=1).tolist()]).T)
                J = np.argmax(label_out, axis=1)
                J1 = np.argmax(label, axis=1)
                counter += np.count_nonzero(J-J1)
                #err_cr -= np.sum(np.array(list(map(lambda x: list(map(lambda y: y[0]*y[1], zip(x[0], x[1]))), zip(label, np.log(label_out))))))
                err_cr -= np.sum(np.multiply(label, np.log(label_out)))
                J = J.tolist()
                ans.append(J)
            err = err_cr/test_num_batches
            if self.Path:
                datafile = self.Path + test_name
                if not os.path.exists(datafile):
                    os.mkdir(datafile)
                datafile += '\\test_result.txt'
                with open(datafile, 'at') as fp:
                    fp.write('%s,%s\n' % (counter, err))
            print(' %s, wrong: %s, error: %s' % (test_name, counter, err))
            print(ans)

        return [ans, counter, err]

    def _logistic(self, x):
        # return 1.0 / (1 + np.exp(-x))
        return .5 * (1 + np.tanh(.5*x))

    def _classify_init(self, w_class, hidden_probs, label):
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
        #label_out = np.array(list(map(lambda x: list(map(lambda y: y/np.sum(x), x)), label_out)))
        label_out = np.divide(label_out, np.array([np.sum(label_out, axis=1).tolist()]).T)
        #f = - np.sum(np.array(list(map(lambda x: list(map(lambda y: y[0]*y[1], zip(x[0], x[1]))), zip(label, np.log(label_out))))))
        f = - np.sum(np.multiply(label, np.log(label_out)))
        IO = label_out - label
        df = np.dot(hidden_probs.T, IO)
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
        hidden_probs = [0] * (self.layers+1) # 0 data 1
        hidden_probs[0] = np.insert(data, 0, 1, axis=1)
        for i in range(0, self.layers):
            hidden_activations = np.dot(hidden_probs[i], weights[i])
            hidden_probs[i+1] = self._logistic(hidden_activations)
            hidden_probs[i+1][:, 0] = 1
        label_out = np.exp(np.dot(hidden_probs[self.layers], w_class))
        #label_out = np.array(list(map(lambda x: list(map(lambda y: y/np.sum(x), x)), label_out)))
        #f = - np.sum(np.array(list(map(lambda x: list(map(lambda y: y[0]*y[1], zip(x[0], x[1]))), zip(label, np.log(label_out))))))
        label_out = np.divide(label_out, np.array([np.sum(label_out, axis=1).tolist()]).T)
        f = - np.sum(np.multiply(label, np.log(label_out)))
        IO = label_out - label
        dw_class = np.dot(hidden_probs[self.layers].T, IO)
        tmp1 = np.dot(IO, w_class.T)
        #tmp2 = np.array(list(map(lambda x: list(map(lambda y: 1-y, x)), hidden_probs[self.layers])))
        tmp2 = np.subtract(1,hidden_probs[self.layers])
        #Ix = np.array(list(map(lambda x: list(map(lambda y: y[0]*y[1]*y[2], zip(x[0], x[1], x[2]))), zip(tmp1, hidden_probs[self.layers], tmp2))))
        Ix = np.multiply(np.multiply(tmp1, hidden_probs[self.layers]), tmp2)
        dw = [0] * (self.layers + 1)
        dw[0] = dw_class
        for i in range(0, self.layers):
            dw[self.layers-i] = np.dot(hidden_probs[self.layers-1-i].T, Ix)
            if i < self.layers - 1:
                tmp1 = np.dot(Ix, weights[self.layers-1-i].T)
                #tmp2 = np.array(list(map(lambda x: list(map(lambda y: 1-y, x)), hidden_probs[self.layers-1-i])))
                tmp2 = np.subtract(1,hidden_probs[self.layers-1-i])
                #Ix = list(map(lambda x: list(map(lambda y: y[0]*y[1]*y[2], zip(x[0], x[1], x[2]))), zip(tmp1, hidden_probs[self.layers-1-i], tmp2)))
                Ix = np.multiply(np.multiply(tmp1, hidden_probs[self.layers-1-i]), tmp2)
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
        INT = 0.1
        EXT = 3.0
        MAX = 20
        RATIO = 10
        SIG = 0.1
        RHO = SIG / 2.0
        i = 0
        Is_failed = 0
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
            X0 = x.copy()
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
                    X0 = list(map(lambda x: x[0] + x3 * x[1], zip(x,s)))
                    F0 = f3
                    dF0 = df3.copy()
                d3 = sum(list(map(lambda x: np.sum(np.multiply(x[0], x[1])), zip(df3, s))))
                if d3 > SIG*d0 or f3 > (f0 + x3 * RHO * d0) or M == 0:
                    break
                x1 = x2
                f1 = f2
                d1 = d2
                x2 = x3
                f2 = f3
                d2 = d3
                A = 6 * (f1 - f2) + 3 * (d2 + d1) * (x2 - x1)
                B = 3 * (f2 - f1) - (2 * d1 + d2) * (x2 - x1)
                x3 = x1 - d1 * ((x2 - x1) ** 2.0) / (B + ((B * B - A * d1 * (x2 - x1)) ** (1/2.0)))
                if (not np.isreal(x3)) or (np.isnan(x3)) or (np.isinf(x3)) or x3 < 0:
                    x3 = x2 * EXT
                else:
                    if x3 > x2 * EXT:
                        x3 = x2 * EXT
                    else:
                        if x3 < (x2 + INT * (x2 - x1)) :
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
                    x3 = x2-(0.5 * d2 * ((x4 - x2) ** 2)) / (f4 - f2 - d2 * (x4 - x2))
                else:
                    A = 6 * (f2 - f4) / (x4 - x2) + 3 * (d4 + d2)
                    B = 3 * (f4 - f2) - (2 * d2 + d4) * (x4 - x2)
                    x3 = x2 + ((B * B - A * d2 * ((x4 - x2) ** 2)) ** (1/2.0) - B) / A
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
                    X0 = list(map(lambda x: x[0] + x3 * x[1], zip(x, s)))
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
                s = list(map(lambda x: (u33 - u03)/u00 * x[0] - x[1], zip(s, df3)))
                df0 = df3.copy()
                d3 = d0
                d0 = sum(list(map(lambda x: np.sum(np.multiply(x[0], x[1])), zip(df0, s))))
                if d0 > 0:
                    s = list(map(lambda x: - x, df0))
                    d0 = -sum(list(map(lambda x: np.sum(np.multiply(x, x)), s)))
                realmin = np.finfo(np.double).tiny
                x3 = x3 * min(RATIO, (d3 / (d0 - realmin)))
                Is_failed = 0
            else:
                x = X0.copy()
                f0 = F0
                df0 = dF0.copy()
                if Is_failed or i > length:
                    break
                s = list(map(lambda x: - x, df0))
                d0 = -sum(list(map(lambda x: np.sum(np.multiply(x, x)), s)))
                x3 = 1 / (1 - d0)
                Is_failed = 1
        return x

# 多个RBM组合类
class RBM_each:
    def __init__(self, num_visible, num_hidden, learning_rate=0.1, Path=None):
        """
        Because we only recognize 10 numbers, so the RBM_each consists of 10 RBMs
        :param num_visible: 可见层单元个数，the number of visible units
        :param num_hidden: 隐含层单元个数，the number of hidden units
        :param learning_rate: 学习率，the learning rate of RBM
        :param Path:   所有RBM参数存储的路径，以//结尾的字符串
                        the path where we store the parameters of RBM, and it ends with //
        """
        self.num_hidden = num_hidden
        self.num_visible = num_visible
        self.learning_rate = learning_rate
        self.path = Path
        self.rbms = []
        for i in range(0, 10):
            tmppath = Path
            tmppath += ('rbm-%d' % i)
            os.mkdir(tmppath)
            tmppath += '\\'
            r = RBM(num_visible=num_visible, num_hidden=num_hidden, learning_rate=learning_rate, Path=tmppath)
            self.rbms.append(r)

    def train(self, train_data, pieces=100, max_epochs=50):
        """
        训练函数 Train Function
        :param train_data:  训练集，类型为list，有10个元素，对应10个数字，
                             元素为np.array, np.array是矩阵，每一行为每一个训练样本
                             training data, type: list of np.array,
                             every np.array is a matrix
                                        where each row is a training example consisting of the states of visible units.
                             i.e. each np.array is a training set of a class
        :param pieces:  每个训练集要分成batches进行训练，每个batches含有的样本数为pieces
                         the number of training example in one batch of a training set of a class
        :param max_epochs: 训练最大迭代次数, the max epochs of the training operation
        """
        for i in range(0, 10):
            batch_data = np.array_split(train_data[i], train_data[i].shape[0]/pieces)
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
        ans=[]
        for item in test:
            minerror = 0
            tmpans = 0
            tmpitem = item.copy()
            tmpitem = [tmpitem]
            for number in range(0, 10):
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

# 读取数据类
class ReadData:
    def __init__(self, dim):
        """
        Initialize Function
        初始化函数
        :param dim: 标签的维度，取值只有10或4
        """
        self.dim = dim

    def read(self, Path, trainflag=0, mnistflag=0, type=1, level=0):
        # 0 784，答案为dim维度  1 784+dim 答案为数字   -1  分类，答案为数字
        """
        读取数据函数
        :param Path: 数据所在路径
        :param trainflag: 是否为训练集
        :param mnistflag: 是否为MNIST数据
        :param type: 读取类型
                0  读出数据784维，标签为dim维度
                1  读出数据为784+dim维度（即图像+标签）， 标签为数字
                -1 读出数据为784维度，且按照标签分类，标签为数字
        :return: list，第1个元素为读出数据，第2个元素为标签
        """
        set = []
        set_ans = []
        if mnistflag:
            mndata = MNIST(Path)
            if trainflag:
                if type < 0:
                    set = [[], [], [], [], [], [], [], [], [], []]
                dataset = mndata.load_training()
                for item in range(0, len(dataset[0])):
                    item_hd = dataset[0][item]
                    item_hd = self.pro(item_hd)
                    if type > 0:
                        self.resize(item_hd, ans=dataset[1][item])
                        set.append(item_hd)
                    else:
                        if type == 0:
                            set.append(item_hd)
                            tmpans = self.resize([], ans=dataset[1][item])
                            set_ans.append(tmpans)
                        else:
                            set[dataset[1][item]].append(item_hd)
                            set_ans.append(dataset[1][item])
                if type < 0:
                    for i in range(0, 10):
                        set[i] = np.array(set[i])
                print("Load TrainingSet Successfully")
            else:
                dataset = mndata.load_testing()
                for item in range(0, len(dataset[0])):
                    item_hd = dataset[0][item]
                    item_hd = self.pro(item_hd)
                    if type > 0:
                        for i in range(0, self.dim):
                            item_hd.append(0)
                        set_ans.append(dataset[1][item])
                        set.append(item_hd)
                    else:
                        if type == 0:
                            set.append(item_hd)
                            tmpans = self.resize([], ans=dataset[1][item])
                            set_ans.append(tmpans)
                        else:
                            set.append(np.array([item_hd]))
                            set_ans.append(dataset[1][item])
                print("Load MNISTTestSet Successfully")
        else:
            if trainflag:
                if type < 0:
                    set = [[], [], [], [], [], [], [], [], [], []]
                import os
                fig = [x for x in os.listdir(Path) if os.path.isfile(Path + x) and x.endswith('png')]
                for f in fig:
                    im = list(Image.open(Path + f).convert('L').getdata())
                    item_hd = self.pro(im)
                    h = len(f)
                    if type > 0:
                        self.resize(item_hd, ans=int(f[h - 5]))
                        set.append(item_hd)
                    else:
                        if type == 0:
                            set.append(item_hd)
                            tmpans = self.resize([], ans=int(f[h - 5]))
                            set_ans.append(tmpans)
                        else:
                            set[int(f[h - 5])].append(item_hd)
                            set_ans.append(int(f[h - 5]))
                if type < 0:
                    for i in range(0, 10):
                        set[i] = np.array(set[i])
                print("Load TrainingSet Successfully")
            else:
                import os
                fig = [x for x in os.listdir(Path) if os.path.isfile(Path + x) and x.endswith('png')]
                for f in fig:
                    if int(f[2]) == level:
                        im = list(Image.open(Path + f).convert('L').getdata())
                        item_hd = self.pro(im)
                        if type > 0:
                            for i in range(0, self.dim):
                                item_hd.append(0)
                            set.append(item_hd)
                            set_ans.append(int(f[0]))
                        else:
                            if type == 0:
                                set.append(item_hd)
                                tmpans = self.resize([], ans=int(f[0]))
                                set_ans.append(tmpans)
                            else:
                                set.append(np.array([item_hd]))
                                set_ans.append(int(f[0]))
                print("Load TestSet Successfully")
        return [set, set_ans]

    def resize(self, item, ans):
        """
        重构标签函数
        用dim维表示标签后加入list中
        :param item: 需要扩展的list
        :param ans: 标签的值
        :return: 扩展后的list
        """
        if self.dim == 10:
            for i in range(0, 10):
                if i == ans:
                    item.append(1)
                else:
                    item.append(0)
        else:
            if self.dim == 4:
                label = bin(ans)
                label = label[::-1]
                length = len(label)
                for i in range(0, length - 3 + 1):
                    item.append(int(label[i]))
                if length != 6:
                    for i in range(1, 6 - length + 1):
                        item.append(0)
        return item

    def makebatches(self, train_data, train_label=None, pieces=100):
        """
        分块数据函数
        :param train_data:  要分块的数据，list，每个元素是list表示一个样本
        :param train_label:  数据标签，list，每个元素对应train_data
        :param pieces: 每块数据的大小，the size of one batch
        :return:
            当有train_label输入时，输出list，
                        第1个元素为list，元素为np.array，每个array是一个batch的data
                        第2个元素为list, 元素为list，该list元素为标签，对应np.array每一行数据
            当无train_label输入时，输出list
                        第1个元素为list，元素为np.array，每个array是一个batch的data
        """
        if train_label == None:
            training_array = np.array(train_data)
            if training_array.shape[0] % pieces == 0:
                batch_num = training_array.shape[0] / pieces
                batch_data = np.split(training_array, batch_num)
                batch_data = [batch_data]
            else:
                batch_num = round(training_array.shape[0] / pieces)
                batch_data = np.array_split(training_array, batch_num)
                batch_data = [batch_data]
        else:
            total = list(map(lambda x: x[0] + x[1], zip(train_data, train_label)))
            total = np.array(total)
            if total.shape[0] % pieces == 0:
                batch_num = total.shape[0] / pieces
                batch_total = np.split(total, batch_num)
            else:
                batch_num = round(total.shape[0] / pieces)
                batch_total = np.array_split(total, batch_num)
            length = batch_total[0].shape[1]
            batch_data = list(map(lambda x: np.array(list(map(lambda y: y[0:length - 10], x.tolist()))), batch_total))
            batch_label = list(map(lambda x: list(map(lambda y: y[length - 10:length], x.tolist())), batch_total))
            batch_data = [batch_data, batch_label]
        return batch_data

    def pro(self, item):
        m = max(item)
        result = list(map(lambda x: x / m, item))
        return result

if __name__ == '__main__':
    method = 1

    dim = 10
    layers = 2
    num_hidden = [784, 500, 2000, 10]
    learning_rate = 0.1
    pieces = 100
    max_epochs_1 = 50
    max_epochs_2 = 200

    mnistPath = 'D:\\Download\\学习\\模式识别\\大作业\\data\\train'

    directory1 = ['D:\\Download\\学习\\模式识别\\大作业\\data\\test\\Process2\\0\\',
                 'D:\\Download\\学习\\模式识别\\大作业\\data\\test\\Process2\\1\\',
                 'D:\\Download\\学习\\模式识别\\大作业\\data\\test\\Process2\\2\\',
                 'D:\\Download\\学习\\模式识别\\大作业\\data\\test\\Process2\\3\\',
                 'D:\\Download\\学习\\模式识别\\大作业\\data\\test\\Process2\\4\\'
                 ]
    directory2 = ['D:\\Download\\学习\\模式识别\\大作业\\data\\test\\Process4\\0\\',
                 'D:\\Download\\学习\\模式识别\\大作业\\data\\test\\Process4\\1\\',
                 'D:\\Download\\学习\\模式识别\\大作业\\data\\test\\Process4\\2\\',
                 'D:\\Download\\学习\\模式识别\\大作业\\data\\test\\Process4\\3\\',
                 'D:\\Download\\学习\\模式识别\\大作业\\data\\test\\Process4\\4\\'
                 ]
    directory = [directory1, directory2]

    rd = ReadData(dim=dim)
    set = rd.read(Path=mnistPath, trainflag=1, mnistflag=1, type=0)
    trainset = set[0]
    labelset = set[1]
    batch_set = rd.makebatches(train_data=trainset, train_label=labelset, pieces=pieces)
    batch_data = batch_set[0]
    batch_label = batch_set[1]

    set = rd.read(Path=mnistPath, trainflag=0, mnistflag=1, type=0)
    testset = set[0]
    labelset = set[1]
    batch_set = rd.makebatches(train_data=testset, train_label=labelset, pieces=pieces)
    batch_test = batch_set[0]
    batch_test_label = batch_set[1]

    test_set = [batch_test]
    test_label_set = [batch_test_label]
    test_name_set = ['MNIST_test']
    for reverseflag in range (0, 2):
        for file in range(0, 5):
            figPath = directory[reverseflag][file]
            for level in range(0, 6):
                if file == 0 and level > 0:
                    break
                set = rd.read(Path=figPath, trainflag=0, mnistflag=0, type=0, level=level)
                if len(set[0]) > 0:
                    testset = set[0]
                    labelset = set[1]
                    batch_test_core = [np.array(testset)]
                    batch_test_label_core = [labelset]
                    test_set.append(batch_test_core)
                    test_label_set.append(batch_test_label_core)
                    test_name = 'Test'+ ('_%d' % file) + ('_%d' % level) + ('_%d' % reverseflag)
                    test_name_set.append(test_name)
                    print('read file %d level %d done' % (file, level))
            print('read file %d done' % file)
        print('read file %d done' % reverseflag)

    path = 'D:\\Download\\学习\\模式识别\\大作业\\Result\\method_'+('%d' % method)+'\\rst'
    path += ('-%dd' % dim)
    path += ('-%dl' % layers)
    path += ('-%dp' % pieces)
    path += ('-%dE' % max_epochs_1)
    path += ('-%de' % max_epochs_2)
    import os
    if not os.path.exists(path):
        os.mkdir(path)
    path += '\\'
    if path:
        datafile = path+'log.txt'
        with open(datafile, 'wt') as fp:
            fp.write(' method:%s\n dim: [ ' % method)
            for i in range(0, len(num_hidden)):
                fp.write('%d ' % num_hidden[i])
            fp.write(']\n learning rate:%s\n piece size:%s\n epochs_1:%s\n epochs_2:%s\n' % (learning_rate, pieces, max_epochs_1, max_epochs_2))


    r = RBM_net(layers=layers, dim=num_hidden, learning_rate=learning_rate, Path=path, mode=1)

    r.train_rbms(batch_data=batch_data, batch_label=batch_label, max_epochs_1=max_epochs_1, max_epochs_2=max_epochs_2)

    print("Train RBM-Net Successfully")

    test_num = len(test_set)
    for i in range(0, test_num):
        r.predict(batch_test=test_set[i], batch_test_label=test_label_set[i], test_name=test_name_set[i])
