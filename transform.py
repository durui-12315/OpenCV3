# -*- coding:UTF-8 -*-
# !/usr/bin/python

"""图像的基本变形：缩放、旋转、裁剪、填充、平移、翻转、仿射变换矩阵"""
import numpy as np
import cv2

# 缩放 scale
# img = cv2.imread("img/0.png")
# img1 = cv2.resize(src=img, dsize=(600, 400), interpolation=cv2.INTER_NEAREST)  # 指定缩放大小和缩放算法
# cv2.imshow("scale1", img1)
# img2 = cv2.resize(src=img, dsize=(0, 0), fx=0.5, fy=0.3)  # 指定缩放比例
# cv2.imshow("scale2", img2)
# cv2.waitKey(10000)

# 旋转 rotate
# img = cv2.imread("img/0.png")
# h, w = img.shape[:2]
# center = (w // 3, h // 2)  # 旋转基准点
# r_mat = cv2.getRotationMatrix2D(center=center, angle=45, scale=0.5)  # 旋转矩阵，缩0.5
# size = (int(w * 1), int(h * 1))  # 输出大小
# r_img = cv2.warpAffine(src=img, M=r_mat, dsize=size, borderValue=(255, 255, 255))  # 仿射变换，白底
# cv2.imshow("rotate", r_img)
# cv2.waitKey(10000)

# 裁剪 crop
# img = cv2.imread("img/0.png")
# h, w = img.shape[:2]
# img = img[h // 4:h // 4 * 3, w // 4:w // 4 * 3]  # 截取数组
# cv2.imshow("crop", img)
# cv2.waitKey(10000)

# 填充 pad
# img = cv2.imread("img/0.png")
# pad_img1 = cv2.copyMakeBorder(src=img, top=100, bottom=100, left=100, right=100,
#                              borderType=cv2.BORDER_REPLICATE)  # 复制法，复制最边缘像素
# cv2.imshow("rotate1", pad_img1)
# pad_img2 = cv2.copyMakeBorder(src=img, top=100, bottom=100, left=100, right=100,
#                               borderType=cv2.BORDER_REFLECT_101)  # 对称法，以最边缘像素为轴，对称
# cv2.imshow("rotate2", pad_img2)
# pad_img3 = cv2.copyMakeBorder(src=img, top=100, bottom=100, left=100, right=100,
#                               borderType=cv2.BORDER_CONSTANT,
#                               value=(100, 200, 200))  # 常量法，以一个常量填充扩充的边界
# cv2.imshow("rotate3", pad_img3)
# cv2.waitKey(10000)

# 平移 translate
# img = cv2.imread("img/0.png")
# h, w = img.shape[:2]
# t_mat = np.float32([[1, 0, -50], [0, 1, 100]])  # 平移矩阵：[[1, 0, 左右],[0, 1, 上下]]
# t_img = cv2.warpAffine(src=img, M=t_mat, dsize=(w, h), borderMode=cv2.BORDER_CONSTANT,
#                        borderValue=(0, 100, 100))  # 左移50，下移100
# cv2.imshow("translate", t_img)
# cv2.waitKey(10000)

# 翻转 flip
# img = cv2.imread("img/0.png")
# horiz_img = cv2.flip(src=img, flipCode=1)  # Y轴翻转
# cv2.imshow("flip-Y", horiz_img)
# verti_img = cv2.flip(src=img, flipCode=0)  # X轴翻转
# cv2.imshow("flip-X", verti_img)
# horandver_img = cv2.flip(src=img, flipCode=-1)  # 水平180度翻转，先X后Y
# cv2.imshow("flip-XY", horandver_img)
# cv2.waitKey(10000)


# 仿射变换矩阵(A * {column, row} + B)实现缩放、旋转、倾斜
img = cv2.imread("img/0.png")
h, w = img.shape[:2]

# 缩放并平移
trans_mat = np.float32([[2, 0, -500], [0, 0.8, 100]])
img1 = cv2.warpAffine(src=img, M=trans_mat, dsize=(w, h), borderMode=cv2.BORDER_CONSTANT,
                      borderValue=(0, 255, 255))
cv2.imshow("affine1", img1)

# 顺时针旋转45度，近似中心旋转
theta, dis_x, dis_y = 45 * np.pi / 180, 0.6 * np.power(2, -0.5) * w, -0.5 * np.power(2, -0.5) * h
trans_mat = np.float32([
    [np.cos(theta), -np.sin(theta), dis_x], [np.sin(theta), np.cos(theta), dis_y]
])
img2 = cv2.warpAffine(src=img, M=trans_mat, dsize=(w, h), borderMode=cv2.BORDER_CONSTANT,
                      borderValue=(255, 255, 0))
cv2.imshow("affine2", img2)

# 45度倾斜
theta, dis_x = 45 * np.pi / 180, -0.5 * np.power(2, -0.5) * w
trans_mat = np.float32([[1, np.tan(theta), dis_x], [0, 1, 0]])
img3 = cv2.warpAffine(src=img, M=trans_mat, dsize=(w, h), borderMode=cv2.BORDER_CONSTANT,
                      borderValue=(0, 255, 255))
cv2.imshow("affine3", img3)
cv2.waitKey(30000)
