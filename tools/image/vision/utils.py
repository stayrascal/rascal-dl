import numpy as np
from PIL import Image
import scipy.signal as signal


def im_conv(image, conv_core):
    image_array = image.copy()
    dim1, dim2 = image_array.shape
    for i in range(1, dim1 - 1):
        for j in range(1, dim2 - 1):
            image_array[i, j] = [image[(i - 1):(i + 2), (j - 1):(j + 2)] * conv_core]
        image_array = image_array * (255.0 / image_array.max())
        return image_array


prewitt_conv_core_x = np.array([[-1, 0, 1],
                                [-1, 0, 1],
                                [-1, 0, 1]])

prewitt_conv_core_y = np.array([[-1, -1, -1],
                                [0, 0, 0],
                                [1, 1, 1]])

sobel_conv_core_x = np.array([[-1, 0, 1],
                              [-2, 0, 2],
                              [-1, 0, 1]])

sobel_conv_core_y = np.array([[-1, -2, -1],
                              [0, 0, 0],
                              [1, 2, 1]])

laplace_conv_core = np.array([[0, 1, 0],
                              [1, -4, 1],
                              [0, 1, 0]])

laplace_conv_core_extend = np.array([[1, 1, 1],
                                     [1, -8, 1],
                                     [1, 1, 1]])


# create Gauss Operator
def func(x, y, sigma=1):
    return 100 * (1 / (2 * np.pi * sigma)) * np.exp(-((x - 2) ** 2 + (y - 2) ** 2) / (2.0 * sigma ** 2))


def detect_edge(img_path):
    gauss_conv_core = np.fromfunction(func, (5, 5), sigma=5)

    # 打开图像并转化为灰度图像
    image = Image.open(img_path).convert('L')
    image_array = np.array(image)

    # 利用生成的高斯算子与原图进行卷积对图像进行平滑处理
    image_blur = signal.convolve2d(image_array, gauss_conv_core, mode="same")

    # 对平滑后的图像进行边缘检测
    image2 = signal.convolve2d(image_blur, laplace_conv_core_extend, mode="same")

    image2 = (image2 / float(image2.max())) * 255

    # 将🐠灰度平均值的灰度值变成255，便于观察边缘
    image2[image2 > image2.mean()] = 255
