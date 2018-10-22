#!/usr/bin/env python

import numpy as np
import cv2
import argparse
from matplotlib import pyplot as plt   #for histogram

# a = np.asarray([[1,3,4,5,7,8],[1,2,4,5,5,7],[1,5,4,2,2,1],[1,3,6,5,3,2]])
# print a
# Y = a.reshape(2,2,-1,2).swapaxes(1,2).reshape(-1, 2, 2)
# print Y.shape
# for i in range(0,6):
# 	print Y[i]

# a = np.asarray([[144,139,149,155,153,155,155,155],
# 	[151,151,151,159,156,156,156,158],
# 	[151,156,160,162,159,151,151,151],
# 	[158,163,161,160,160,160,160,161],
# 	[158,160,161,162,160,155,155,156],
# 	[161,161,161,161,160,157,157,157],
# 	[162,162,161,160,161,157,157,157],
# 	[162,162,161,160,163,157,158,154]])
# print a
# print cv2.dct(np.float32(a))

# Should print
# 1257.9      2.3     -9.7     -4.1      3.9      0.6     -2.1      0.7 
#  -21.0    -15.3     -4.3     -2.7      2.3      3.5      2.1     -3.1 
#  -11.2     -7.6     -0.9      4.1      2.0      3.4      1.4      0.9 
#   -4.9     -5.8      1.8      1.1      1.6      2.7      2.8     -0.7 
#    0.1     -3.8      0.5      1.3     -1.4      0.7      1.0      0.9 
#    0.9     -1.6      0.9     -0.3     -1.8     -0.3      1.4      0.8 
#   -4.4      2.7     -4.4     -1.5     -0.1      1.1      0.4      1.9 
#   -6.4      3.8     -5.0     -2.6      1.6      0.6      0.1      1.5 

# a = np.asarray([[[1,2,3],[3,4,5]],[[4,5,6],[7,8,9]]])
# print a[0]
# print a[1]
# print np.mean(a,axis = 0)
# print np.mean(a,axis = 1)
# print np.mean(a,axis = 2)

# a = np.asarray([1+2j,2+3j,3+4j])
# print np.absolute(a)

# x = np.arange(0,200)
# print np.sin(30*2*np.pi*x/200), x.shape
# print np.random.randn(len(x))

# y = np.sin(30*2*np.pi*x/200)+np.random.randn(len(x))

# plt.plot(y)
# plt.show()

# z = np.absolute(np.fft.fft(y))

# plt.plot(z)
# plt.show()
firstq = 30
secondq = 40
for x in range(0,20):
	for y in range(0,2):
		if y == 0:
			image = cv2.imread("../dataset/ucid00070.tif")
			cv2.imwrite("../dataset/comp"+str(firstq)+".jpg",image,[int(cv2.IMWRITE_JPEG_QUALITY),firstq])
		if y==1:
			image = cv2.imread("../dataset/comp"+str(firstq)+".jpg")
			cv2.imwrite("../dataset/comp"+str(firstq)+str(secondq)+".jpg",image,[int(cv2.IMWRITE_JPEG_QUALITY),secondq])
# cv2.imwrite("../dataset/comp"+str(firstq)+str(secondq)+".jpg",image,[int(cv2.IMWRITE_JPEG_QUALITY),secondq])
	firstq += 5
	secondq += 5
# cv2.imwrite("../comp90.jpg",image,[int(cv2.IMWRITE_JPEG_QUALITY),90])
