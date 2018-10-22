#!/usr/bin/env python

import numpy as np
from scipy import fftpack as fftp
import cv2
import argparse
from matplotlib import pyplot as plt   #for histogram
import csv
import pandas as pd

 #=============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
firstq = 30
secondq = 40
file = open("data2.csv",'a')
for x in range(0,20):
     for y in range(0,2):
         thres = 0.5
         if y == 0:
             image = cv2.imread("../dataset/comp"+str(firstq)+".jpg")
         if y==1:
             image = cv2.imread("../dataset/comp"+str(firstq)+str(secondq)+".jpg")
         if y==0:
             text = str(firstq) + "," + str(0) + "," 
         if y ==1 :
             text = str(firstq) + "," + str(secondq) + "," 
         dct_rows = 0;
         dct_cols = 0;
         print (text)
         shape = image.shape;
     
         if shape[0]%8 != 0:
         	dct_rows = shape[0]+8-shape[0]%8;
         else:
         	dct_rows = shape[0];
         if shape[1]%8 != 0:
         	dct_cols = shape[1]+8-shape[1]%8;
         else:
         	dct_cols = shape[1];	
         dct_image = np.zeros((dct_rows,dct_cols,3),np.uint8)
         dct_image[0:shape[0], 0:shape[1]] = image
         reserve_text = text
         for h in range(0,5):
             text = reserve_text + str(thres) + ","
             #print (thres)
     			# image = cv2.imread("../dataset/comp7090.jpg")
     			# firstq = 70
     			# secondq = 90
             y = cv2.cvtColor(dct_image, cv2.COLOR_BGR2YCR_CB)[:,:,0]
             
             w=y.shape[1]
             h=y.shape[0]
             n = w*h/64
             #print n," 8x8 blocks"
     
             Y = y.reshape(h//8,8,-1,8).swapaxes(1,2).reshape(-1, 8, 8)
     			#print Y.shape, Y[0].shape
     
     			
     			#print Q_mat
     
             qDCT =[]
             for i in range(0,Y.shape[0]):
                 qDCT.append(cv2.dct(np.float32(Y[i])))
     
             qDCT = np.asarray(qDCT, dtype=np.float32)
     			#print qDCT.shape
     
             qDCT = np.rint(qDCT - np.mean(qDCT, axis = 0)).astype(np.int32)
     			#print qDCT.shape
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
             #print (a1)
             for idx,ax in enumerate(a1):
                 #print(idx/8)
                 data = qDCT[:,int(idx/8),int(idx%8)]
                 val,key = np.histogram(data, bins=np.arange(data.min(), data.max()+1),normed = True)
                 z = np.absolute(fftp.fft(val))
                 z = np.reshape(z,(len(z),1))
                 rotz = np.roll(z,int(len(z)/2))
     				# denoisedz = (np.roll(rotz,-2)+np.roll(rotz,-1)+rotz+np.roll(rotz,1)+np.roll(rotz,2))/5
     				#print val.shape, key.shape, z.shape
     				
     				# print val
     				# print z
                 slope = rotz[1:] - rotz[:-1]
                 indices = [i+1 for i in range(len(slope)-1) if slope[i] > 0 and slope[i+1] < 0]
     				# print indices
                 peak_count = 0
     
     				
                 for j in indices:
     					#print rotz[j][0]
                     if rotz[j][0]>thres:
                         peak_count+=1
     				#print peak_count
                 text+= str(peak_count) +","
     				# thres+=0.1 
     				
     				# print num_peaks
                 ax.plot(rotz)
     			#print text
             #print (text)
             plt.ioff()
             plt.close()
             file.write(text+"\n")
             thres+=0.1
     firstq += 5
     secondq += 5
     file.close()
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================


#getting threshold plots
# =============================================================================
df = pd.read_csv("data2.csv")
# 
# 
thresh_pd = 0.5
for x in range(0,5):
    value_list = [round(thresh_pd,1)]
    print (value_list)
    value_list_q2 = [0]
    new_df = df[df.thresh.isin(value_list)]
    single_new_df = df[df.q2.isin(value_list_q2)]
    double_new_df = df[~df.q2.isin(value_list_q2)]
    single_array = np.array(single_new_df[['2', '3','4','9','10','11','17','18','25']])
    double_array = np.array(double_new_df[['2', '3','4','9','10','11','17','18','25']])
    f2,a2 = plt.subplots(3,3)
    a2 = a2.ravel()
    for idx,ax in enumerate(a2):
        plot_value_single = single_array[:,idx]
        plot_value_double = double_array[:,idx]
        ax.plot(plot_value_single,'r')
        ax.plot(plot_value_double,'b')
         
         
    print (single_array)
    print (double_array)
    thresh_pd+=0.1
# =============================================================================
    
    
    
