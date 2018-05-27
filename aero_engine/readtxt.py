#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 读取txt文件里的传感器数据
import os
import sys
import numpy as np
from skimage import io,transform

train_path = "/Users/lijian/Desktop/tensorflow_code/aero_engine/train_sensor.txt"

#将所有的图片重新设置尺寸为32*32
w = 32
h = 32
c = 1

#读取图片及其标签函数
def read_image(path):
	f = open(path)
	images = []
	labels = []
	line = f.readline()
	while line:
		list1 = line.split(" ")
		list2 = list1[:256]
		image = [float(item) for item in list2]
		label = list1[-2:-1]
		label = float(list1[306])

		image = np.array(image) #list转为array才能reshape
		image = image.reshape((16,16))
		image = transform.resize(image,(w,h,c)) #将所有的图片重新设置尺寸为32*32*1
		images.append(image)
		labels.append(label)
		line = f.readline()
		#print("image is:",image)
	f.close()
	return np.asarray(images,dtype=np.float32),np.asarray(labels,dtype=np.int32)


train_data,train_label = read_image(train_path)

print("image is:",train_data)
print(len(train_data))
print(type(train_data))
print("label is:",train_label)


