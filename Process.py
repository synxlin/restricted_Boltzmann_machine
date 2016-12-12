import numpy as np
from scipy import ndimage
from PIL import Image
from numpy import *
from scipy.misc import imsave

def Reconstruct (im, U_init, tolerance=0.2, tau=0.1, tv_weight=10):
    m, n = im.shape
    # 初始化
    U = U_init
    Px = im # 对偶域的x 分量
    Py = im # 对偶域的y 分量
    error = 1
    while (error > tolerance):
      Uold = U
      # 原始变量的梯度
      GradUx = roll(U, -1, axis=1)-U # 变量U 梯度的x 分量
      GradUy = roll(U, -1, axis=0)-U # 变量U 梯度的y 分量
      # 更新对偶变量
      PxNew = Px + (tau/tv_weight)*GradUx
      PyNew = Py + (tau/tv_weight)*GradUy
      NormNew = maximum(1, sqrt(PxNew**2+PyNew**2))
      Px = PxNew/NormNew # 更新x 分量（对偶）
      Py = PyNew/NormNew # 更新y 分量（对偶）
      # 更新原始变量
      RxPx = roll(Px,1,axis=1) # 对x 分量进行向右x 轴平移
      RyPy = roll(Py,1,axis=0) # 对y 分量进行向右y 轴平移
      DivP = (Px-RxPx)+(Py-RyPy) # 对偶域的散度
      U = im + tv_weight*DivP # 更新原始变量
      # 更新误差
      error = linalg.norm(U-Uold)/sqrt(n*m)
    return U

def process (im, fig_name):
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
    U = Reconstruct(G, M)
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
            end_row =31
    label_im = label_im[start_row:end_row+1, start_column:end_column+1]
    imsave('D:\\Download\\学习\\模式识别\\大作业\\data\\2.png', label_im)
    baseim = Image.open('D:\\Download\\学习\\模式识别\\大作业\\data\\1.png')
    floatim = Image.open('D:\\Download\\学习\\模式识别\\大作业\\data\\2.png')
    floatim = floatim.resize((19, 19), Image.LANCZOS)
    baseim.paste(floatim, (5, 5))
    baseim.save('D:\\Download\\学习\\模式识别\\大作业\\data\\test\\Process2\\'+fig_name)
    return baseim

directory = ['D:\\Download\\学习\\模式识别\\大作业\\data\\test\\noise2\\0\\',
             'D:\\Download\\学习\\模式识别\\大作业\\data\\test\\noise2\\1\\',
             'D:\\Download\\学习\\模式识别\\大作业\\data\\test\\noise2\\2\\',
             'D:\\Download\\学习\\模式识别\\大作业\\data\\test\\noise2\\3\\',
             'D:\\Download\\学习\\模式识别\\大作业\\data\\test\\noise2\\4\\'
             ]

for item in range(0, 5):
    file = directory[item]
    h = len(file)
    import os
    fig = [x for x in os.listdir(file) if os.path.isfile(file+x) and x.endswith('png') ]
    for f in fig:
        im = Image.open(file+f).convert('L')
        figname = file[h-2:] + f
        im = process(im, figname)
