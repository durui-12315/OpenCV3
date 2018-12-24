# -*- coding:UTF-8 -*-
# !/usr/bin/python

import numpy as np
import cv2
import os
import time

# 静态图像人脸检测
# def detect(img_path):
#     face_cascade = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")  # 分类器
#     img = cv2.imread(img_path)
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)  # 在灰度图像上人脸检测(多个坐标点)
#     for x, y, w, h in faces:  # 根据坐标点在彩图上画矩形
#         img = cv2.rectangle(img=img, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
#     cv2.namedWindow("face")
#     cv2.imshow("face", img)
#     cv2.waitKey(10000)
#
# detect("img/ice&fire2.jpg")


# 动态图像检测（摄像头检测人脸模型）
cascade_file_hash = {
    "face": cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml"),
    "l_eye": cv2.CascadeClassifier("haarcascade/haarcascade_mcs_lefteye_alt.xml"),
    "r_eye": cv2.CascadeClassifier("haarcascade/haarcascade_mcs_righteye_alt.xml"),
    "mouth": cv2.CascadeClassifier("haarcascade/haarcascade_mcs_mouth.xml"),
    "nose": cv2.CascadeClassifier("haarcascade/haarcascade_mcs_nose.xml")
}
face_label_hash = {1: "rex", 2: "iris"}
face_label_hash_ = {"rex": 1, "iris": 2}

# 0.人脸检测demo
# def face_detect(face_frame, cascade_file):
#     gray = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY)
#     # 脸部轮廓
#     faces_coord = cascade_file["face"].detectMultiScale(gray, 1.3, 5)
#     for x, y, w, h in faces_coord:
#         face_frame = cv2.rectangle(face_frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
#     # # 左眼睛
#     # leye_coord = cascade_file["l_eye"].detectMultiScale(gray, 1.3, 5)
#     # for x, y, w, h in leye_coord:
#     #     face_frame = cv2.rectangle(face_frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 255), thickness=2)
#     # # 右眼睛
#     # reye_coord = cascade_file["r_eye"].detectMultiScale(gray, 1.3, 5)
#     # for x, y, w, h in reye_coord:
#     #     face_frame = cv2.rectangle(face_frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 255), thickness=2)
#     # # 嘴巴
#     # reye_coord = cascade_file["mouth"].detectMultiScale(gray, 1.3, 5)
#     # for x, y, w, h in reye_coord:
#     #     face_frame = cv2.rectangle(face_frame, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
#     # 鼻子
#     reye_coord = cascade_file["nose"].detectMultiScale(gray, 1.3, 5)
#     for x, y, w, h in reye_coord:
#         face_frame = cv2.rectangle(face_frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=2)
#     return face_frame
#
# cc = cv2.VideoCapture(0)
# cv2.namedWindow("face", cv2.WINDOW_AUTOSIZE)
# success, frame = cc.read()  # 读取摄像头图像
# while success:
#     frame = face_detect(frame, cascade_file_hash)  # 检测人脸
#     cv2.imshow("face", frame)
#     keyCode = cv2.waitKey(1)
#     if keyCode == 27:
#         break
#     success, frame = cc.read()  # 继续读取新图像
# cv2.destroyAllWindows()
# cc.release()  # 释放资源

def make_dir(path):
    if os.path.isdir(path):
        for f in os.listdir(path): os.remove(os.path.join(path, f))
    else:
        os.mkdir(path)

def time_name(path):
    name = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    return '%s\\%s.jpg' % (path, name)

# 1.生成人脸识别数据
# name, count = 'iris', 0
# path = 'img\\face\\%s' % name
# make_dir(path)
# cc = cv2.VideoCapture(0)
# cv2.namedWindow("face", cv2.WINDOW_AUTOSIZE)
# success, frame = cc.read()
# while success:
#     frame = cv2.flip(frame, flipCode=1)  # Y轴翻转
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     coord = cascade_file_hash["face"].detectMultiScale(gray, 1.3, 5)
#     if len(coord):  # 如果有检测结果
#         count += 1
#         x, y, w, h = coord[0][0], coord[0][1], coord[0][2], coord[0][3]  # 第一人脸矩形参数
#         frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         if count % 10 == 0:  # 不连续保存
#             frame_scale = gray[x:x + w, y:y + h]  # 头像部分
#             frame_scale = cv2.resize(frame_scale, (200, 200), interpolation=cv2.INTER_NEAREST)  # 统一大小
#             cv2.imwrite(path + '\\%d.jpg' % (count // 10), frame_scale)  # 保存图像
#     cv2.imshow("face", frame)
#     keyCode = cv2.waitKey(1)
#     if keyCode == 27: break
#     success, frame = cc.read()
# cv2.destroyAllWindows()
# cc.release()

# 2.训练数据准备
# # 摄像头数据
# dirnames = os.listdir('img\\face')
# with open("face_train1.csv", "w+") as f:
#     for dirs in dirnames:
#         dirname = 'img\\face\\' + dirs
#         if os.path.isdir(dirname):
#             files = os.listdir(dirname)
#             for file in files:
#                 filename = '%s\\%s' % (dirname, file)
#                 if os.path.isfile(filename): f.writelines("%s;%d\n" % (filename, face_label_hash_[dirs]))
# # 人脸图片数据
# lst, count = [], 0
# for dirname, dirnames, filenames in os.walk("img\\att_faces"):  # 广度优先遍历原理
#     for subdir in dirnames:
#         sublst, subpath = [], os.path.join(dirname, subdir)
#         for f in os.listdir(subpath):
#             sublst.append(os.path.join(subpath, f))
#         lst.append(sublst)
#     break
# with open("face_train.csv", "w+") as f:
#     for sub in lst:
#         for i in sub:
#             f.writelines(i + ";%d\n" % count)
#         count += 1

# 3.加载训练数据
face_mat, face_label = [], []
with open("face_train1.csv", 'r') as f:
    for i in f.readlines():
        i_ = i.split(';')
        if os.path.exists(i_[0]):
            i_[1] = i_[1].replace('\n', '')
            i_[0] = cv2.imread(i_[0], cv2.IMREAD_GRAYSCALE)
            i_[0] = cv2.resize(i_[0], (200, 200))
            face_mat.append(i_[0])
            face_label.append(int(i_[1]))
face_mat = np.asarray(face_mat, dtype=np.uint8)
face_label = np.asarray(face_label, dtype=np.int32)

# 4.人脸识别训练与预测(threshold为置信度阈值)
# 训练
model = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8, threshold=123.0)
model.train(face_mat, face_label)
# 测试
cc = cv2.VideoCapture(0)
cv2.namedWindow("face", cv2.WINDOW_AUTOSIZE)
success, frame = cc.read()
while success:
    frame = cv2.flip(frame, flipCode=1)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_coord = cascade_file_hash["face"].detectMultiScale(gray_frame, 1.3, 5)
    for x, y, w, h in faces_coord:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        frame = cv2.circle(frame, center=(x, y), radius=4, color=(0, 0, 255), thickness=2)
        frame = cv2.circle(frame, center=(x + w, y + h), radius=4, color=(0, 0, 255), thickness=2)
        roi = gray_frame[x:x + w, y:y + h]
        roi = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_LINEAR)
        param = model.predict(roi)
        if param[0] != -1:
            title = '%s: %d' % (face_label_hash[param[0]], param[1])
            cv2.putText(img=frame, text=title, org=(x, y - 20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 0, 255), thickness=2)
    cv2.imshow("face", frame)
    keyCode = cv2.waitKey(1)
    if keyCode == 27:
        break
    elif keyCode == 13:
        cv2.imwrite(time_name('img\\face'), frame)
    success, frame = cc.read()
cv2.destroyAllWindows()
cc.release()
