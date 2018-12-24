# -*- coding:UTF-8 -*-
# !/usr/bin/python

import numpy as np
import cv2
from scipy import ndimage

# np.set_printoptions(threshold=np.inf)

# 高通滤波器(根据像素和周围临近像素的差值提升像素的亮度) 和 低通滤波器(差值小于特定值时，平滑该像素的亮度)
# kernel_3x3 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.int8)
# kernel_5x5 = np.array([
#     [-1, -1, -1, -1, -1], [-1, 1, 2, 1, -1], [-1, 2, 4, 2, -1], [-1, 1, 2, 1, -1], [-1, -1, -1, -1, -1]
# ], dtype=np.int8)
# img = cv2.imread("img/0.jpg", cv2.IMREAD_GRAYSCALE)
# k3 = ndimage.convolve(img, kernel_3x3)  # 卷积函数，高通滤波
# k5 = ndimage.convolve(img, kernel_5x5)
# blur = cv2.GaussianBlur(img, ksize=(11, 11), sigmaX=0)  # 高斯模糊（最常用的低通滤波），奇数越大越模糊
# g_hpf = img - blur  # 差值后为高通滤波
# cv2.imshow("k3", k3)
# cv2.imshow("k5", k5)
# cv2.imshow("blur", blur)
# cv2.imshow("g_hpf", g_hpf)
# cv2.waitKey(10000)
# cv2.destroyAllWindows()

# 边缘检测
# img = cv2.imread("img/0.jpg")
# blur = cv2.medianBlur(img, 5)  # 中值模糊
# gray_blur = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)  # 彩色变灰色，反之COLOR_GRAY2BGR
# cv2.Laplacian(gray_blur, cv2.CV_8U, gray_blur, ksize=5)  # 边缘检测，黑色背景
# cv2.imshow("blur", gray_blur)
# cv2.imshow("blur1", 255 - gray_blur)  # 白色背景
# cv2.imshow("blur2", (255 - gray_blur) / 255 * gray_blur)  # 归一化 和 矩阵元素积
# cv2.waitKey(20000)

# 卷积滤波器
# img = cv2.imread("img/0.jpg")
# kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.int8)
# cv2.filter2D(img, -1, kernel, img)  # -1表示源图像和目标图像都有同样的位深度
# cv2.imshow("convolution", img)
# cv2.waitKey(200000)

# canny边缘检测
# img = cv2.imread("img/0.jpg", cv2.COLOR_BGR2GRAY)
# cv2.GaussianBlur(img, (5, 5), 0, img)  # 先高斯模糊
# cv2.imshow("canny1", cv2.Canny(img, 200, 300))  # 再canny边缘化，效果好
# cv2.waitKey(20000)

# 轮廓检测
# img = np.zeros((600, 600), dtype=np.uint8)
# img[150:450, 150:450] = 255
# ret, thresh = cv2.threshold(img, 127, 255, 0)
# image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # contours为矩形
# color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# img = cv2.drawContours(color_img, contours, -1, (0, 255, 0), 2)  # 在图片上画轮廓
# cv2.imshow("edge", color_img)
# cv2.waitKey(20000)

# 轮廓检测(不规则、矩形、圆)
# img = cv2.imread("img/apple.jpg")
# ret, thresh = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 200, 255, cv2.THRESH_BINARY)
# image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# for c in contours:
#     # 1
#     x, y, w, h = cv2.boundingRect(c)  # 得到矩形轮廓坐标
#     if w > 10 and h > 10:
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 根据轮廓坐标得到矩形并绘制
#     # 2
#     rect = cv2.minAreaRect(c)  # 最小矩形区域
#     if rect[1][0] > 10 and rect[1][1] > 10:
#         box = cv2.boxPoints(rect)  # 计算最小区域坐标
#         box = np.int0(box)  # 转整
#         cv2.drawContours(img, [box], 0, (0, 0, 255), 3)  # 绘制边框
#     # 3
#     (x, y), radius = cv2.minEnclosingCircle(c)  # 得到最小闭圆半径和中心点坐标
#     if radius > 10:
#         center, radius = (int(x), int(y)), int(radius)  # 转整
#         cv2.circle(img, center, radius, (0, 255, 0), 2)  # 绘制圆形轮廓
# cv2.drawContours(img, contours, -1, (255, 0, 0), 1)  # 所有轮廓(不规则)
# cv2.imshow("contours", img)
# cv2.waitKey(20000)

# 直线检测
# img = cv2.imread("img/0.jpg")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray, 50, 120)
# minLineWidth, maxLineGap = 20, 5
# lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineWidth, maxLineGap)  # 直线检测函数，在轮廓图里检测
# for x1, y1, x2, y2 in lines[:, 0, :]:
#     length = int(pow(pow(x1 - x2, 2) + pow(y1 - y2, 2), 0.5))
#     if length > 10:
#         cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 画在原图上
# cv2.imshow("edges", edges)
# cv2.imshow("lines", img)
# cv2.waitKey(20000)

# 圆检测
img = cv2.imread("img/planet.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur_img = cv2.medianBlur(gray, 5)
cimg = cv2.cvtColor(blur_img, cv2.COLOR_GRAY2BGR)
c = cv2.HoughCircles(blur_img, cv2.HOUGH_GRADIENT, 1, 120, param1=100, param2=30, minRadius=0, maxRadius=0)
c = np.uint16(np.round(c))
for i in c[0, :]:
    cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)  # 画圆边
    cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)  # 圆心
cv2.imshow("circle", img)
cv2.waitKey(20000)
