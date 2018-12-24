# -*- coding:UTF-8 -*-
# !/usr/bin/python

import numpy as np
import cv2, sys

'''
图像金字塔建立：
1.使用任意尺度的参数缩小原图像大小；
2.平滑图像（高斯模糊）；
3.如果缩小平滑的图像比大于最小尺寸，则重复上述过程，直到小于或等于

非最大值抑制（NMS）：
1.作用：消除多余交叉重叠的窗口，找到目标最佳检测位置
2.过程：
    （1）建立图像金字塔，用滑动窗口搜索当前层的图像目标；
    （2）收集所有含有目标的窗口，得到分类概率最大的窗口W；
    （3）消除所有与W的IOU重叠度大于某个阈值的窗口；
    （4）移动到下一个分类概率最大的窗口W1，在当前尺度下重复上述过程
    （5）上述过程完成后，在图像金字塔的下一个尺度继续重复上述过程
    
词袋（BOW）：用来一系列文档中计算每歌词出现的次数，用这些次数构成向量来重新表示文档
'''

# HOG目标检测
# img = cv2.imread('img\\group_photo.jpg')
# hog = cv2.HOGDescriptor()
# hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())  # SVM检测人形
# found, w = hog.detectMultiScale(img)
# found_filter = []
# for x1, y1, w1, h1 in found:
#     for x2, y2, w2, h2 in found:
#         if x1 > x2 and y1 > y2 and x1 + w1 < x2 + w2 and y1 + h1 < y2 + h2: break  # break执行时不会执行else
#     else:
#         found_filter.append([x1, y1, w1, h1])
# for x, y, w, h in found_filter:
#     img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
# cv2.imshow('person', img)
# cv2.waitKey(10000)

# 汽车检测 数据集：http://ai.stanford.edu/~jkrause/cars/car_dataset.html
# 图像金字塔
def pyramid(image, scale=1.5, min_size=(200,80)):
    while image.shape[0] >= min_size[1] and image.shape[1] >= min_size[0]:
        yield image
        w = int(image.shape[1] / scale)
        h = int(image.shape[0] / scale)
        image = cv2.resize(image, dsize=(w, h), interpolation=cv2.INTER_AREA)

# 滑动窗口
def sliding_window(image, step=(200, 80), window_size=(50, 20)):
    for row in range(0, image.shape[0] - window_size[1], step[1]):
        for col in range(0, image.shape[1] - window_size[0], step[0]):
            yield (row, col, image[row:row + window_size[1], col:col + window_size[0]])

# 非极大值抑制1
def non_maximum_suppression_fast(bounding_boxes, overlapThresh):
    if len(bounding_boxes) == 0: return []
    # 更改数据类型为浮点
    if bounding_boxes.dtype.kind == 'u' or bounding_boxes.dtype.kind == 'i':
        bounding_boxes = bounding_boxes.astype('float')
    # 初始筛选列表 & 拆分数组值
    pick, s = [], bounding_boxes[:, 4]
    x1, y1 = bounding_boxes[:, 0], bounding_boxes[:, 1]
    x2, y2 = bounding_boxes[:, 2], bounding_boxes[:, 3]
    # 面积 & 分值排序
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(s)[::-1]  # 倒置，分值从大到小
    while np.size(idxs) > 0:
        # 最小值下标
        last = np.size(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # 找到与其它矩形的重合区域的宽高，非重合置0
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # 重合区域与对应矩形的重合率
        overlap = (w * h) / area[idxs[:last]]
        # 删除超过阈值的矩形
        filter_id = np.where(overlap > overlapThresh)[0]
        idxs = np.delete(idxs, np.concatenate(([last], filter_id), axis=0))
    return bounding_boxes[pick].astype('int')

# 非极大值抑制2
def non_maximum_suppression(bounding_boxes, overlapThreth):
    if np.size(bounding_boxes) == 0 or overlapThreth > 1:
        return np.int([])
    # 转类型
    if bounding_boxes.dtype.kind == 'i' or bounding_boxes.dtype.kind == 'u':
        bounding_boxes = bounding_boxes.astype('float')
    # 值拆分
    pick, s = [], bounding_boxes[:, -1]
    x1, y1 = bounding_boxes[:, 0], bounding_boxes[:, 1]
    x2, y2 = bounding_boxes[:, 2], bounding_boxes[:, 3]
    # 按分值降序排序，得到下标（指针思想）
    idxs = np.argsort(-s, axis=0, kind='quicksort')
    while np.size(idxs) > 0:
        # 当前最大分值下标和面积
        i = idxs[0]
        i_area = (x2[i] - x1[i] + 1) * (y2[i] - y1[i] + 1)
        pick.append(i)
        # 计算重合区域坐标及宽高，非重合置0
        xx1, yy1 = np.maximum(x1[idxs], x1[i]), np.maximum(y1[idxs], y1[i])
        xx2, yy2 = np.minimum(x2[idxs], x2[i]), np.minimum(y2[idxs], y2[i])
        w, h = np.maximum((xx2 - xx1 + 1), 0), np.maximum((yy2 - yy1 + 1), 0)
        # 计算与当前最大分值重合率
        overlap = (w * h) / i_area
        # 去掉重合率大于阈值的下标
        filter_idxs = np.where(overlap >= overlapThreth)[0]  # 直接返回idxs下标
        idxs = np.delete(idxs, filter_idxs, axis=0)
    return bounding_boxes[pick].astype('int')

# 目标展示
def detect_show(image, bounding_boxes):
    for x1, y1, x2, y2, ss in bounding_boxes:
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
        image = cv2.putText(image, '%.5f' % ss, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    cv2.imshow('detect', image)
    cv2.waitKey(30000)


if __name__ == '__main__':
    # 非极大值抑制测试
    # img_h, img_w, nums, max_w, max_h = 600, 1000, 50, 200, 200
    # x = np.random.randint(0, img_w - max_w - 10, size=(nums,1), dtype=np.uint16)
    # y = np.random.randint(0, img_h - max_h - 10, size=(nums,1), dtype=np.uint16)
    # w = np.random.randint(30, max_w, size=(nums,1), dtype=np.uint16)
    # h = np.random.randint(30, max_h, size=(nums,1), dtype=np.uint16)
    # s = np.random.randint(1, 500, size=(nums,1), dtype=np.uint16)
    # boxes = np.concatenate((x, y, x + w, y + h, s), axis=1)
    # # boxes = non_maximum_suppression_fast(bounding_boxes=boxes, overlapThresh=0.1)
    # boxes = non_maximum_suppression(bounding_boxes=boxes, overlapThreth=0.1)
    # img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    # i = 0
    # for x1, y1, x2, y2, ss in boxes:
    #     img = cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
    #     img = cv2.putText(img, '%d:%d' % (i, ss), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    #     i += 1
    # cv2.imshow('nms', img)
    # cv2.waitKey(90000)

    # 非深度学习目标检测流程（训练预测省略）
    image = cv2.imread('img\\0.jpg')
    boxes, window_size = [], (200, 80)
    for img in pyramid(image, scale=1.1):
        if img.shape[0] < window_size[1] or img.shape[1] < window_size[0]:
            break
        scale = image.shape[1] / img.shape[1]
        for y, x, roi in sliding_window(img, step=(100, 40), window_size=window_size):
            score = np.random.rand() * 100  # 假设为预测概率
            if score > 80:  # 概率达到阈值，按比例映射原图并加入bounding box中
                x1, y1 = int(x * scale), int(y * scale)
                x2, y2 = int((x + window_size[0]) * scale), int((y + window_size[1]) * scale)
                boxes.append([x1, y1, x2, y2, score])
    boxes = non_maximum_suppression(bounding_boxes=np.float32(boxes), overlapThreth=0.1)  # 非极大值抑制
    detect_show(image, boxes)  # 目标检测展示




