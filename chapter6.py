# -*- coding:UTF-8 -*-
# !/usr/bin/python

import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt

# Harris角点检测
# img = cv2.imread('img\\chess.jpg')
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_gray = np.float32(img_gray)  # 转浮点
# dst = cv2.cornerHarris(src=img_gray, blockSize=2, ksize=23, k=0.04)
# img[dst > 0.01 * dst.max()] = [0, 0, 255]  # 通过布尔索引重新赋值
# cv2.imshow('corner', img)
# cv2.waitKey(10000)

# SURF特征检测 & 命令行运行sys.argv[0]为python脚本名
# img = cv2.imread(sys.argv[1])
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# if sys.argv[2] == 'SURF':
#     fd = cv2.xfeatures2d.SURF_create(float(sys.argv[3]) if len(sys.argv) == 4 else 4000)
# else:
#     fd = cv2.xfeatures2d.SIFT_create()
# keypoints, descriptor = fd.detectAndCompute(gray_img, None)
# img = cv2.drawKeypoints(img, keypoints=keypoints, outImage=img, color=(0, 255, 0),
#                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imshow('SURF', img)
# cv2.waitKey(30000)

# ORB特征匹配
# logo = cv2.imread('img\\ferrari_logo.jpg', cv2.IMREAD_GRAYSCALE)
# car = cv2.imread('img\\ferrari_car.jpg', cv2.IMREAD_GRAYSCALE)
# orb = cv2.ORB_create()
# kp_logo, des_logo = orb.detectAndCompute(logo, None)  # 特征检测，可通过np.save保存description
# kp_car, des_car = orb.detectAndCompute(car, None)  # 特征检测
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# matches = bf.match(des_logo, des_car)  # 特征匹配，返回最佳匹配
# matches = sorted(matches, key=lambda x: x.distance)
# img = cv2.drawMatches(logo, kp_logo, car, kp_car, matches[:40], car, flags=2)  # 描绘前40个匹配结果
# plt.imshow(img)
# plt.show()

# KNN K-最近邻匹配
logo = cv2.imread('img\\ferrari_logo.jpg', cv2.IMREAD_GRAYSCALE)
car = cv2.imread('img\\ferrari_car.jpg', cv2.IMREAD_GRAYSCALE)
orb = cv2.ORB_create()
kp_logo, des_logo = orb.detectAndCompute(logo, None)  # 特征检测
kp_car, des_car = orb.detectAndCompute(car, None)  # 特征检测
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.knnMatch(des_logo, des_car, k=2)  # KNN特征匹配，返回k个匹配
matches = sorted(matches, key=lambda x: x.distance)
img = cv2.drawMatchesKnn(logo, kp_logo, car, kp_car, matches, car, flags=2)  # 描绘匹配结果
plt.imshow(img)
plt.show()

