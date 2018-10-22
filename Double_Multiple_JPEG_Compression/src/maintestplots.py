#!/usr/bin/env python

import numpy as np
from scipy import fftpack as fftp
import cv2
import argparse
from matplotlib import pyplot as plt   #for histogram


# image = cv2.imread("../dataset/ucid00070.tif")
# image = cv2.imread("../dataset/comp90.jpg")
image = cv2.imread("../dataset/comp9070.jpg")
y = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)[:,:,0]

w=y.shape[1]
h=y.shape[0]
n = w*h/64
print n," 8x8 blocks"

Y = y.reshape(h//8,8,-1,8).swapaxes(1,2).reshape(-1, 8, 8)
print Y.shape, Y[0].shape

Q_mat = np.asarray([[16, 11, 10, 16, 24, 40, 51, 61],
	[12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77], 
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]], dtype=np.int32)
print Q_mat

qDCT =[]
for i in range(0,Y.shape[0]):
	qDCT.append(cv2.dct(np.float32(Y[i])))

qDCT = np.asarray(qDCT, dtype=np.float32)
print qDCT.shape

qDCT = np.rint(qDCT - np.mean(qDCT, axis = 0)).astype(np.int32)
print qDCT.shape
# print qDCT[11]

# f,a = plt.subplots(8,8)
# a = a.ravel()
# for idx,ax in enumerate(a):
# 	data = qDCT[:,idx/8,idx%8]
# 	ax.hist(data,bins=np.arange(data.min(), data.max()+1),normed =True)
# plt.tight_layout()
# plt.show()

f,a1 = plt.subplots(8,8)
a1 = a1.ravel()
for idx,ax in enumerate(a1):
	data = qDCT[:,idx/8,idx%8]
	val,key = np.histogram(data, bins=np.arange(data.min(), data.max()+1),normed = True)
	z = np.absolute(fftp.fft(val))
	z = np.reshape(z,(len(z),1))
	rotz = np.roll(z,len(z)/2)
	# denoisedz = (np.roll(rotz,-2)+np.roll(rotz,-1)+rotz+np.roll(rotz,1)+np.roll(rotz,2))/5

	print val.shape, key.shape, z.shape
	if idx == 0:
		print val
		print z
	# ax.plot(denoisedz)
	ax.plot(rotz)
plt.tight_layout()
plt.show()

cv2.waitKey(0)

cv2.destroyAllWindows()