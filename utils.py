# -*-coding:utf-8-*-
from __future__ import print_function

from numpy import *
from mnist import MNIST
from PIL import Image
from scipy import ndimage
from scipy.misc import imsave


# 读取数字图像数据类
class ReadDigitImageData:
    def __init__(self, dim):
        """
        Initialize Function
        初始化函数
        :param dim: 标签的维度，取值只有10或4 dim of labels, 10 or 4 = log2(10)
        """
        assert dim == 10 or dim == 4
        self.dim = dim

    def read(self, path, trainflag=0, mnistflag=0, output_type=1, noise_level=0):
        # 0 784，答案为dim维度  1 784+dim 答案为数字   -1  分类，答案为数字
        """
        读取数据函数
        :param path: 数据所在路径, path to image data
        :param trainflag: 是否为训练集, is train set
        :param mnistflag: 是否为MNIST数据, is mnist dataset
        :param output_type: 读取类型
                0  读出数据784维，标签为dim维度(list)
                1  读出数据为784+dim维度（即图像+标签）， 标签为数字 (int)
                -1 读出数据为784维度，且按照标签分类，标签为数字(int) (filename的第1个字符(数字))
        :param noise_level: 噪声等级，filename的第2个字符(数字)
        :return: list，第1个元素为读出数据的list of data，第2个元素为标签的list of labels
        """
        if output_type < 0:
            assert self.dim == 10
        set_img = []
        set_ans = []
        if mnistflag:
            mndata = MNIST(path)
            if trainflag:
                if output_type < 0:
                    set_img = [[], [], [], [], [], [], [], [], [], []]
                dataset = mndata.load_training()
                for item in range(0, len(dataset[0])):
                    item_hd = dataset[0][item]
                    item_hd = self.pro(item_hd)
                    if output_type > 0:
                        self.resize(item_hd, ans=dataset[1][item])
                        set_img.append(item_hd)
                        set_ans.append(dataset[1][item])
                    elif output_type == 0:
                        set_img.append(item_hd)
                        set_ans.append(self.resize([], ans=dataset[1][item]))
                    else:
                        set_img[dataset[1][item]].append(item_hd)
                        set_ans.append(dataset[1][item])
                if output_type < 0:
                    for i in range(0, self.dim):
                        set_img[i] = np.array(set_img[i])
                print("Load Training Set Successfully")
            else:
                dataset = mndata.load_testing()
                for item in range(0, len(dataset[0])):
                    item_hd = dataset[0][item]
                    item_hd = self.pro(item_hd)
                    if output_type > 0:
                        for i in range(0, self.dim):
                            item_hd.append(0)
                        set_img.append(item_hd)
                        set_ans.append(dataset[1][item])
                    else:
                        if output_type == 0:
                            set_img.append(item_hd)
                            tmpans = self.resize([], ans=dataset[1][item])
                            set_ans.append(tmpans)
                        else:
                            set_img.append(np.array([item_hd]))
                            set_ans.append(dataset[1][item])
                print("Load MNIST Test Set Successfully")
        else:
            if trainflag:
                if output_type < 0:
                    set_img = [[], [], [], [], [], [], [], [], [], []]
                fig = [x for x in os.listdir(path) if os.path.isfile(path + x) and x.endswith('png')]
                for f in fig:
                    im = list(Image.open(path + f).convert('L').getdata())
                    item_hd = self.pro(im)
                    if output_type > 0:
                        self.resize(item_hd, ans=int(f[0]))
                        set_img.append(item_hd)
                        set_ans.append(int(f[0]))
                    else:
                        if output_type == 0:
                            set_img.append(item_hd)
                            tmpans = self.resize([], ans=int(f[0]))
                            set_ans.append(tmpans)
                        else:
                            set_img[int(f[0])].append(item_hd)
                            set_ans.append(int(f[0]))
                if output_type < 0:
                    for i in range(0, self.dim):
                        set_img[i] = np.array(set_img[i])
                print("Load Training Set Successfully")
            else:
                fig = [x for x in os.listdir(path) if os.path.isfile(path + x) and x.endswith('png')]
                for f in fig:
                    if int(f[2]) == noise_level:
                        im = list(Image.open(path + f).convert('L').getdata())
                        item_hd = self.pro(im)
                        if output_type > 0:
                            for i in range(0, self.dim):
                                item_hd.append(0)
                            set_img.append(item_hd)
                            set_ans.append(int(f[0]))
                        else:
                            if output_type == 0:
                                set_img.append(item_hd)
                                tmpans = self.resize([], ans=int(f[0]))
                                set_ans.append(tmpans)
                            else:
                                set_img.append(np.array([item_hd]))
                                set_ans.append(int(f[0]))
                print("Load Test Set Successfully")
        return set_img, set_ans

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
            label = bin(ans)
            label = label[::-1]
            length = len(label)
            for i in range(0, length - 3 + 1):
                item.append(int(label[i]))
            if length != 6:
                for i in range(1, 6 - length + 1):
                    item.append(0)
        return item

    @staticmethod
    def make_batches(train_data, train_label=None, batch_size=100):
        """
        分块数据函数
        :param train_data:  要分块的数据，list，每个元素是list表示一个样本
        :param train_label:  数据标签，list，每个元素对应train_data
        :param batch_size: 每块数据的大小，the size of one batch
        :return:
            当有train_label输入时，输出
                        第1个元素为list，元素为np.array，每个array是一个batch的data
                        第2个元素为list, 元素为list，该list元素为标签，对应np.array每一行数据
            当无train_label输入时，输出
                        第1个元素为list，元素为np.array，每个array是一个batch的data
        """
        batch_label = None
        if train_label is None:
            training_array = np.array(train_data)
            if training_array.shape[0] % batch_size == 0:
                batch_num = training_array.shape[0] / batch_size
                batch_data = np.split(training_array, batch_num)
                batch_data = batch_data
            else:
                batch_num = round(training_array.shape[0] / batch_size)
                batch_data = np.array_split(training_array, batch_num)
                batch_data = batch_data
        else:
            total = list(map(lambda x: x[0] + x[1], zip(train_data, train_label)))
            total = np.array(total)
            if total.shape[0] % batch_size == 0:
                batch_num = total.shape[0] / batch_size
                batch_total = np.split(total, batch_num)
            else:
                batch_num = round(total.shape[0] / batch_size)
                batch_total = np.array_split(total, batch_num)
            length = batch_total[0].shape[1]
            batch_data = list(map(lambda x: np.array(list(map(lambda y: y[0:length - 10], x.tolist()))), batch_total))
            batch_label = list(map(lambda x: list(map(lambda y: y[length - 10:length], x.tolist())), batch_total))
        return batch_data, batch_label

    @staticmethod
    def pro(item):
        m = max(item)
        result = list(map(lambda x: x / m, item))
        return result


def reconstruct(im, u_init, tolerance=0.2, tau=0.1, tv_weight=10):
    m, n = im.shape
    # 初始化
    u = u_init
    px = im  # 对偶域的x 分量
    py = im  # 对偶域的y 分量
    error = 1
    while error > tolerance:
        u_old = u
        # 原始变量的梯度
        grad_ux = roll(u, -1, axis=1) - u  # 变量U 梯度的x 分量
        grad_uy = roll(u, -1, axis=0) - u  # 变量U 梯度的y 分量
        # 更新对偶变量
        px_new = px + (tau/tv_weight)*grad_ux
        py_new = py + (tau/tv_weight)*grad_uy
        norm_new = maximum(1, sqrt(px_new**2 + py_new**2))
        px = px_new / norm_new  # 更新x 分量（对偶）
        py = py_new / norm_new  # 更新y 分量（对偶）
        # 更新原始变量
        rx_px = roll(px, 1, axis=1)  # 对x 分量进行向右x 轴平移
        ry_py = roll(py, 1, axis=0)  # 对y 分量进行向右y 轴平移
        div_p = (px-rx_px) + (py-ry_py)  # 对偶域的散度
        u = im + tv_weight*div_p  # 更新原始变量
        # 更新误差
        error = linalg.norm(u-u_old)/sqrt(n*m)
    return u


def process(im, fig_name):
    # 增加对比度
    im = np.array(im)
    im = 255 - im
    im = np.multiply(im, 255/np.max(im))
    im = 255 - im
    # 初步滤波
    im = ndimage.gaussian_filter(im, sigma=0.5)
    im = ndimage.percentile_filter(im, 20, 2)
    # 阈值选择
    a = np.mean(im)
    b = np.min(im)
    W1 = im.shape[0]
    W2 = im.shape[1]
    for i in range(0, W1):
        for j in range(0, W2):
            if im[i, j] > (3*a/4+1*b/4):
                im[i, j] = 255
            else:
                im[i, j] = 0
    # 再次滤波
    M = ndimage.median_filter(im, size=2)
    G = ndimage.gaussian_filter(im, sigma=0.5)
    # 以高斯滤波为含噪输入，中值滤波为初值，重构图像
    U = reconstruct(G, M)
    # 阈值选择 并作 最值滤波
    a = np.mean(U)
    b = np.min(U)
    for i in range(0, W1):
        for j in range(0, W2):
            if U[i, j] > a:
                U[i, j] = 255
    T = ndimage.maximum_filter(U, 1)
    # 二值化后删除连通面积小的区域
    for i in range(0, W1):
        for j in range(0, W2):
            if T[i, j] > 200:
                T[i, j] = 0
            else:
                T[i, j] = 1
    label_im, nb_labels = ndimage.label(T)
    sizes = ndimage.sum(T, label_im, range(nb_labels + 1))
    a = mean(sizes)
    if size(sizes) > 5:
        mask_size = sizes < 2*a
        remove_pixel = mask_size[label_im]
        label_im[remove_pixel] = 0
    else:
        if size(sizes) > 2:
            mask_size = sizes < 0.8 * a
            remove_pixel = mask_size[label_im]
            label_im[remove_pixel] = 0
    # 恢复图像
    a = max(sizes)
    i = 0
    for i in range(0, W1):
        for j in range(0, W2):
            if label_im[i, j] > 0:
                label_im[i, j] = (1-(sizes[label_im[i, j]]/a-1)**2) * 255
            else:
                label_im[i, j] = 0
    for i in range(0, W1):
        if np.sum(label_im[i]) > 0:
            break
    start_row = i
    for i in range(0, W1):
        if np.sum(label_im[W1-1-i]) > 0:
            break
    end_row = W1-1-i
    for i in range(0, W2):
        if sum(label_im.T[i]) > 0:
            break
    start_column = i
    for i in range(0, W2):
        if sum(label_im.T[W2-1-i]) > 0:
            break
    end_column = W2-1-i
    band_row = end_row - start_row + 1
    band_column = end_column - start_column + 1
    if band_row > band_column:
        mid_column = round((start_column + end_column)/2.0)
        start_column = round(mid_column - band_row/2.0)
        end_column = round(mid_column + band_row/2.0-1)
        if start_column < 0:
            start_column = 0
        if end_column > 31:
            end_column = 31
    else:
        mid_row = round((start_row + end_row)/2.0)
        start_row = round(mid_row - band_column/2.0)
        end_row = round(mid_row + band_column/2.0-1)
        if start_row < 0:
            start_row = 0
        if end_row > 31:
            end_row = 31
    label_im = label_im[start_row:end_row+1, start_column:end_column+1]
    imsave('2.png', label_im)
    baseim = Image.open('1.png')
    floatim = Image.open('2.png')
    floatim = floatim.resize((19, 19), Image.LANCZOS)
    baseim.paste(floatim, (5, 5))
    baseim.save(os.path.join('new', fig_name))
    return baseim
