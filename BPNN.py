# -*- coding:UTF-8 -*-
# !/usr/bin/python

"""
    简易BP神经网络：输入层、隐藏层、输出层
    模型优化总结：调整学习率、多批次训练模型、改变网络形状、训练数据广而杂
"""
import numpy as np
import scipy.special as ss
import os, cv2, math
from random import shuffle


# 3层BP神经网络
class BPNN:
    def __init__(self, inputs_num, hidden_num, outputs_num):
        self.__af = lambda x: ss.expit(x)  # 激活函数sigmoid
        self.lr = 0.1
        if not self.check_save():
            self.__iptn = int(inputs_num)
            self.__hidn = int(hidden_num)
            self.__optn = int(outputs_num)
            iptr, hidr = pow(self.__iptn, -0.5), pow(self.__hidn, -0.5)
            self.__wih = np.random.normal(-iptr, iptr, size=(self.__hidn, self.__iptn))  # 输出层权重初始矩阵
            self.__who = np.random.normal(-hidr, hidr, size=(self.__optn, self.__hidn))  # 隐藏层权重初始矩阵
        else:
            self.__wih = np.load("wih.npy")
            self.__who = np.load("who.npy")

    def __to_array(self, lists):
        return np.array(lists, ndmin=2).T

    @staticmethod
    def check_save():
        return os.path.exists("wih.npy") and os.path.exists("who.npy")

    def train(self, input_list, target_list):
        inputs = self.__to_array(input_list)
        hidden_inputs = np.dot(self.__wih, inputs)
        hidden_outputs = self.__af(hidden_inputs)
        final_inputs = np.dot(self.__who, hidden_outputs)
        final = self.__af(final_inputs)
        output_errors = self.__to_array(target_list) - final  # 输出层误差
        hidden_errors = np.dot(self.__who.T, output_errors)  # 隐藏层反向传播的误差
        # 隐藏层权值更新
        self.__who += self.lr * np.dot(output_errors * final * (1 - final), hidden_outputs.T)
        # 输入层权值跟新
        self.__wih += self.lr * np.dot(hidden_errors * hidden_outputs * (1 - hidden_outputs), inputs.T)

    def query(self, input_list):
        inputs = self.__to_array(input_list)
        hidden_inputs = np.dot(self.__wih, inputs)
        hidden_outputs = self.__af(hidden_inputs)
        final_inputs = np.dot(self.__who, hidden_outputs)
        final_outputs = self.__af(final_inputs)
        return np.argmax(final_outputs)  # 下标即数字

    def save(self):
        np.save("wih", self.__wih)
        np.save("who", self.__who)


def random_rotate(records):
    num_mat = np.array(records, dtype=np.uint8).reshape((28, 28))  # 列表转数组时，一定要是正整数
    angle, rnd = 0, np.random.randint(0, 5)
    if rnd == 0:
        angle = -15
    elif rnd == 1:
        angle = 15
    elif rnd == 3:
        angle = -10
    elif rnd == 4:
        angle = 10
    r_mat = cv2.getRotationMatrix2D(center=(14, 14), angle=angle, scale=1.0)  # 旋转矩阵
    num_mat = cv2.warpAffine(src=num_mat, M=r_mat, dsize=(28, 28), borderValue=255)
    return num_mat.reshape(28 * 28)


def transform(img_path):
    num_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).reshape(28 * 28)
    num_img = (255 - num_img) / 255 * 0.98 + 0.01
    return num_img


if __name__ == "__main__":
    hidden_nodes = 196
    if BPNN.check_save():
        bp = BPNN(28 ** 2, hidden_nodes, 10)
        for i in range(10):
            lst = transform(str(i) + ".png")
            print(bp.query(lst), end=' ')
    else:
        training_data_file = open("mnist_train.csv", 'r')
        training_data_list = training_data_file.readlines()
        for k, record in enumerate(training_data_list):  # 数据转换
            record = record.split(',')
            n = int(record[0])
            training_data_list[k] = [int(i) for i in record]
            training_data_list[k][0] = n
        training_data_file.close()

        j = 0
        for i in range(10000):
            num_mat = 255 - cv2.imread(str(j) + ".png", cv2.IMREAD_GRAYSCALE).reshape(28 ** 2)
            num_mat = np.insert(num_mat, 0, j, axis=0)
            training_data_list.append(num_mat.tolist())
            j += 1
            if j == 10: j = 0
        shuffle(training_data_list)

        test_data_file = open("mnist_test.csv", 'r')
        test_data_list = test_data_file.readlines()
        for k, record in enumerate(test_data_list):  # 数据转换
            record = record.split(',')
            n = int(record[0])
            test_data_list[k] = [int(i) / 255 * 0.98 + 0.01 for i in record]  # 归一化
            test_data_list[k][0] = n
        test_data_file.close()

        j = 0
        for i in range(167):
            num_mat = 255 - cv2.imread(str(j) + ".png", cv2.IMREAD_GRAYSCALE).reshape(28 ** 2)
            num_mat = np.insert(num_mat, 0, j, axis=0)
            test_data_list.append(num_mat.tolist())
            j += 1
            if j == 10: j = 0
        shuffle(test_data_list)

        bp = BPNN(28 ** 2, hidden_nodes, 10)
        epochs, total, flag, batch, step = 0, len(test_data_list), 0, 5000, 0
        mini_batch = int(math.ceil(len(training_data_list) / batch))
        while True:
            bp.lr = 0.1 * pow(0.9, epochs)
            epochs += 1
            for j in range(mini_batch):  # 小批量梯度下降
                # 模型训练
                step += 1
                print("Step %d: learning rate %.3f," % (step, bp.lr), end=' ')
                for record in training_data_list[j * batch:(j + 1) * batch]:
                    targets = [0.99 if i == record[0] else 0.01 for i in range(10)]
                    img = random_rotate(record[1:])  # 数据增强（随机旋转）
                    img = img / 255 * 0.98 + 0.01  # 归一化输入
                    bp.train(img, targets)
                # 模型测试
                correct = 0
                for record in test_data_list:
                    index = bp.query(record[1:])
                    if index == record[0]: correct += 1
                # 正确率
                rate = correct / total
                print("correct rate %.5f" % rate)
                if rate > 0.976:
                    bp.save()
                    flag = 1
                    break
            if flag: break

