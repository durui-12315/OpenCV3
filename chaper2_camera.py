# -*- coding:UTF-8 -*-
# !/usr/bin/python
import cv2
import numpy as np
import time


# 时间名字
def time_name(path="img/camera/"):
    name = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    return path + name + ".jpg"


name, stop = "face", False
cc = cv2.VideoCapture(0)
cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
success, frame = cc.read()
while success:  # stop为按鼠标键退出
    cv2.imshow(name, frame)
    keyCode = cv2.waitKey(1)
    if keyCode == 27:  # 按esc键退出
        break
    elif keyCode == 13:  # 按回车键存图片（按时间命名）
        cv2.imwrite(time_name(), frame)
    elif keyCode == 32:  # 按空格键暂停/启动
        stop = not stop
    if not stop:
        success, frame = cc.read()  # 读取新摄像
cv2.destroyAllWindows()
cc.release()


