# -*- coding:UTF-8 -*-
# !/usr/bin/python

"""安装命令：1.pip install opencv-python  2.pip install opencv-contrib-python"""
import cv2
import numpy as np
import os
import time

# 创建黑白图像
# img = np.random.randint(0, 255, size=(300, 300), dtype=np.uint8)  # 随机整数
# cv2.imshow("BW", img)
# img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# print(img.shape)
# cv2.imshow("BGR", img)
# cv2.waitKey(5000)
# print(np.random.binomial(10, 0.1, size=(10, 10)))  # 二项式分布

# 图像格式转换
# img = cv2.imread("img/0.jpg")  # 会删除透明度通道信息
# print(img, img.shape)
# cv2.imwrite("img/0.png", img)  # 要求BGR或灰度格式

# 读取图像为灰度图像
# img = cv2.imread("img/0.png", cv2.IMREAD_GRAYSCALE)  # Gray = (R*19595 + G*38469 + B*7472) >> 16
# print(img, img.shape)
# cv2.imwrite("img/0_.png", img)

# 图像与原始字节转换
# grayImg = np.array(bytearray(os.urandom(300 * 400))).reshape(300, 400)
# cv2.imshow('rndImg', grayImg)
# # colorImg = np.random.randint(0, 255, size=(300, 400, 3), dtype=np.uint8)
# colorImg = grayImg.reshape(100, 400, 3)
# cv2.imshow('colorImg', colorImg)
# cv2.waitKey(10000)

# 使用数组索引访问数据
# imgArr = cv2.imread("img/1.jpg")
# imgArr.itemset((-1, -1, 0), 255)  # 等价
# print(imgArr.item(-1, -1, 0))
# imgArr[:, :, 0] = 0  # 蓝色通道为0
# s = imgArr.shape
# cv2.imshow("part", imgArr[int(s[0] / 4):int(s[0] / 4 * 3), int(s[1] / 4):int(s[1] / 4 * 3), :])
# imgArr[:500, 500:1000] = imgArr[-500:, -1000:-500]
# cv2.imshow("all", imgArr)
# print(imgArr.shape, imgArr.size, imgArr.dtype)
# cv2.waitKey()  # 等价0
# cv2.destroyAllWindows()  # 销毁窗口

# 循环显示图片
# i = 0
# while cv2.waitKey(1) == -1:
#     name = "img/" + str(i) + ".jpg"
#     i += 1
#     if i == 10: i = 0;
#     img = cv2.imread(name)
#     cv2.imshow("hehe", img)  # 名字相同则覆盖原来图片
# cv2.destroyAllWindows()

# 视频的读和写，复制视频（只复制画面）
# vc = cv2.VideoCapture("video/wangyujia.mp4")
# fps = vc.get(cv2.CAP_PROP_FPS)
# size = (int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# vw = cv2.VideoWriter("video/wangyujia_.avi", cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
# success, frame = vc.read()  # 读，success返回布尔，frame返回每帧BGR图像数组
# while success:
#     vw.write(frame)  # 写入新视频
#     success, frame = vc.read()  # 继续读

# 捕获摄像头的帧，视频保存
cc = cv2.VideoCapture(0)
fps = 30  # 帧率 30Hz
size = (int(cc.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cc.get(cv2.CAP_PROP_FRAME_HEIGHT)))
vw = cv2.VideoWriter("video/camera.avi", cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
success, frame = cc.read()
remain = 10 * fps  # 时间限制
while success and remain > 0:
    vw.write(frame)
    success, frame = cc.read()
    remain -= 1
cc.release()  # 释放资源

# 窗口显示摄像头帧
# stop, name, short = False, "face", False
# def do_stop(event, x, y, flags, param):
#     global stop
#     if event == cv2.EVENT_LBUTTONUP:
#         stop = True
#
# cc = cv2.VideoCapture(0)
# cv2.namedWindow(name)
# cv2.setMouseCallback(name, do_stop)
# print("press any key to stop or click widow")
# success, frame = cc.read()
# while success:  # stop为按鼠标键退出
#     cv2.imshow(name, frame)
#     keyCode = cv2.waitKey(1)
#     if keyCode == 32 or stop:  # 按空格键退出
#         break
#     elif keyCode == 13:  # 按回车键存图片（按时间命名）
#         cv2.imwrite("img/camera/" + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + ".jpg", frame)
#     success, frame = cc.read()  # 读取新摄像
# cv2.destroyAllWindows()
# cc.release()


