import zernikemoments as zmm
import numpy as np
import argparse
import _pickle as cPickle
import math
import glob
import cv2
from PIL import Image
import sys

#sys.stdout = open("euclidean_zernike.doc", "w")

def Euclidean_distance(l1,l2):
	sm=0
	for i in range(0,len(l1)):
		sm=sm+(l1[i]-l2[i])*(l1[i]-l2[i])
	res=math.sqrt(sm)
	#print res
	return res

def check(zer_values,indices):
	avg = 0 # to find avg euclidean distance between matchers
	matched=[]
	matched_seg_ind=[]
	T=0.001 # set threshold
	for i in range(0,len(zer_values)-1):
		j=i+1
		#print Euclidean_distance(zer_values[i],zer_values[j])
		eucl = Euclidean_distance(zer_values[i],zer_values[j])
		avg = avg + eucl
		if  eucl < T :
			matched.append([zer_values[i],zer_values[j]])     # list[[zernike_values_of_1,zernike_value_of_2],....]
			matched_seg_ind.append([i,j])   # pair of indices of matched segments

	len_of_zer = len(zer_values)
	#print "average is: "
	#print float(avg)/float(len_of_zer)

	#print len(matched_seg_ind)
	return matched_seg_ind
