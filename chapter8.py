# -*- coding:UTF-8 -*-
# !/usr/bin/python

import numpy as np
import cv2

# KNN背景分割器
bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)
cc = cv2.VideoCapture(0)
success, frame = cc.read()
while success:
    frame = cv2.flip(frame, 1)
    fgmask = bs.apply(frame)
    th = cv2.threshold(fgmask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
    dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
    img, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) > 1600:
            x, y, w, h = cv2.boundingRect(c)
            frame = cv2.rectangle(frame, (w, y), (x + w, y + h), (255, 255, 0), 2)
    cv2.imshow('mog', fgmask)
    cv2.imshow('therth', th)
    cv2.imshow('detection', frame)
    if cv2.waitKey(1) == 27: break
    success, frame = cc.read()
cv2.destroyAllWindows()
cc.release()


