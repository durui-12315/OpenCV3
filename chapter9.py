# -*- coding:UTF-8 -*-
# !/usr/bin/python

import numpy as np
import cv2
from random import randint as rnd
import sys
from chapter7 import non_maximum_suppression as nms
from BPNN import BPNN

'''
神经网络的学习算法：
    1.监督学习：从神经网络获得一个函数，用来描述标记的数据；
    2.非监督学习：数据没有标记，数据的分类通常由聚类等技术实现；
    3.强化学习：系统接收输入，决策机制决定决策行为，执行该机制并给出相应评分（成功与失败之间）。
最后，输入和动作要与评分相匹配，系统重复学习或根据一定的输入或状态来改变动作。
'''

# 动物分类      输入：weight、length、teeth；输出：dog、eagle、dolphin、dragon
# dog = lambda x=True: np.float32([[rnd(5, 20), 1, rnd(38, 42)] if x else [1, 0, 0, 0]])
# eagle = lambda x=True: np.float32([[rnd(3, 13), 3, 0] if x else [0, 1, 0, 0]])
# dolphin = lambda x=True: np.float32([[rnd(3, 190), rnd(5, 15), rnd(80, 100)] if x else [0, 0, 1, 0]])
# dragon = lambda x=True: np.float32([[rnd(1200, 1800), rnd(15, 40), rnd(110, 180)] if x else [0, 0, 0, 1]])
# nn = cv2.ml.ANN_MLP_create()
# nn.setTrainMethod(cv2.ml.ANN_MLP_RPROP | cv2.ml.ANN_MLP_UPDATE_WEIGHTS)  # 反向传播方式
# nn.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)  # 激活函数
# nn.setLayerSizes(np.array([3, 8, 4]))  # 网络层级
# nn.setTermCriteria((cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1))  # 终止条件
# # 生成数据
# Epoches, Records, Test, record = 5, 5000, 100, []
# for i in range(0, Records):
#     record.append((dog(), dog(False)))
#     record.append((eagle(), eagle(False)))
#     record.append((dolphin(), dolphin(False)))
#     record.append((dragon(), dragon(False)))
# # 迭代训练
# for i in range(1, Epoches + 1):
#     for ipt, opt in record:
#         nn.train(ipt, cv2.ml.ROW_SAMPLE, opt)
# # 模型测试
# test = {k: 0 for k in ['dog', 'eagle', 'dolphin', 'dragon']}
# for i in range(Test):
#     if nn.predict(dog())[0] == 0.0: test['dog'] += 1
#     if nn.predict(eagle())[0] == 1.0: test['eagle'] += 1
#     if nn.predict(dolphin())[0] == 2.0: test['dolphin'] += 1
#     if nn.predict(dragon())[0] == 3.0: test['dragon'] += 1
# for k, v in test.items():
#     print('%s: %.2f%%' % (k, v / 100))

# 手写数字识别
# # 训练和测试数据存储
# train_data = []
# with open('../algrithm/BPNN/mnist_test.csv') as f:
#     for read in f.readlines():
#         read = read.split(',')
#         train_data.append(read)
# train_data = np.asfarray(train_data, dtype=np.float32)
# train_data[:, 1:] = train_data[:, 1:] / 255 * 0.98 + 0.01
# np.save('mnist_test', train_data)
# 数据读取
train_data, test_data = np.load('mnist_train.npy'), np.load('mnist_test.npy')
# 模型建立
nn = cv2.ml.ANN_MLP_create()
nn.setLayerSizes(np.float32([28 * 28, 128, 10]))
nn.setTrainMethod(cv2.ml.ANN_MLP_RPROP)
nn.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
nn.setTermCriteria((cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 1))
# 模型训练
target = np.ones(shape=(train_data.shape[0], 10), dtype=np.float32) * 0.01
target[[i for i in range(train_data.shape[0])], train_data[:, 0].astype('int')] = 0.99  # 花式索引改值
batch, step, epoch = 60000, 0, 0
end = np.ceil(train_data.shape[0] / batch) * batch
while True:
    epoch += 1
    nn.train(train_data[step:step + batch, 1:], cv2.ml.ROW_SAMPLE, target[step:step + batch])
    step += batch
    if step == end: step = 0
    # 模型预测
    predict = nn.predict(test_data[:, 1:], cv2.ml.ROW_SAMPLE)
    result = np.argmax(predict[1], axis=1) == test_data[:, 0]
    result = result.mean()
    print('Epoch %d: %.1f%%' % (epoch, result * 100))
    if result > 0.85: break

# 使用模型
# 1.图片颜色处理和轮廓标记
# bp = BPNN(28**2, 196, 10)
img = cv2.imread('img\\number1.jpg')
bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bw = cv2.GaussianBlur(bw, (7, 7), 0)
ret, thbw = cv2.threshold(bw, 127, 255, cv2.THRESH_BINARY_INV)  # 黑底白字，逆二进制阈值
thbw = cv2.erode(thbw, np.ones(shape=(2, 2), dtype=np.uint8), iterations=2)  # 平滑化
image, cntrs, hier = cv2.findContours(thbw.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 目标轮廓
# 2.去掉嵌套的轮廓
bboxes, boxes = [], []
for c in cntrs:
    a = cv2.contourArea(c)  # 面积
    if a < 5: continue
    x, y, w, h = cv2.boundingRect(c)  # 轮廓
    bboxes.append([x, y, x + w, y + h, a])
bboxes = sorted(bboxes, key=lambda x: x[4], reverse=True)  # 面积降序
for x1, y1, x2, y2, a in bboxes:
    inside = False
    for x1_, y1_, x2_, y2_, a_ in boxes:
        if x1 > x1_ and y1 > y1_ and x2 < x2_ and y2 < y2_:  # 检查是否大包小
            inside = True
            break
    if not inside: boxes.append([x1, y1, x2, y2, a])
# 3.轮廓中的数字原比例转成28*28的小数字
numbers = np.zeros(shape=(1, 28 ** 2), dtype=np.uint8)
for x1, y1, x2, y2, a in boxes:
    number = np.zeros(shape=(28, 28), dtype=np.uint8)
    num, x, y = thbw[y1:y2, x1:x2], x2 - x1, y2 - y1
    if y > x:
        num_ = np.zeros(shape=(y, y), dtype=np.uint8)
        dis = int((y - x) / 2)
        num_[:, dis:dis + x] = num
    else:
        num_ = np.zeros(shape=(x, x), dtype=np.uint8)
        dis = int((x - y) / 2)
        num_[dis:dis+y, :] = num
    num_ = cv2.resize(num_, dsize=(24, 24))
    number[2:-2, 2:-2] = num_
    # cv2.imshow('%d'%np.random.randint(1, 10000), number)
    numbers = np.concatenate((numbers, number.reshape(1, 28 ** 2)), axis=0)  # 行拼接
numbers = np.delete(numbers, 0, axis=0)  # 删除第一行
# 4.数字预测并标记原图
predict = nn.predict(numbers / 255 * 0.98 + 0.01, cv2.ml.ROW_SAMPLE)
predict = np.argmax(predict[1], axis=1)
for i, number in enumerate(numbers):
    # predict = bp.query(number / 255 * 0.98 + 0.01)
    img = cv2.putText(img, str(predict[i]), org=(boxes[i][0], boxes[i][1] - 10),
                  fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 0, 255), thickness=2)
# 5.原图展示结果
for x, y, x1, y1, area in boxes:
    cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 2)
img[(img[:, :, 0] == 255) & (img[:, :, 1] == 255) & (img[:, :, 2] == 255)] -= 10  # 纯白底加深一点点
cv2.imshow('number', img)
cv2.waitKey(20000)
