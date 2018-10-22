from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import math
from PIL import Image

def check(a1,a2,b1,b2):
	
	# D=60 if image size>=3000*2000 else D=30

	dis=(a1-b1)*(a1-b1)+(a2-b2)*(a2-b2)	
	dis=math.sqrt(dis)
	#print dis
	return dis


"""def Euclidean_distance(l1,l2):
	sm=0
	for i in range(0,len(l1)):
		sm=sm+(l1[i]-l2[i])*(l1[i]-l2[i])
	res=math.sqrt(sm)
	#print "h "
	#print res
	return res

"""

def filter_outliers(matching_pairs,final,zernike_values):
	avg = 0
	filtered=[]
	D = 60
	for i in range(0,len(matching_pairs)):
		ind1=matching_pairs[i][0]
		ind2=matching_pairs[i][1]
		a1 = final[ind1][0]
		a2 = final[ind1][1]
		b1 = final[ind2][0]
		b2 = final[ind2][1]
		distance = check(a1,a2,b1,b2)
		#print distance 
		if(distance > D):
			#print distance
			#print Euclidean_distance(zernike_values[ind1],zernike_values[ind2])
			#avg = avg + Euclidean_distance(zernike_values[ind1],zernike_values[ind2])
			filtered.append([a1,a2,b1,b2])	
	#print avg
	#len_fil = len(filtered)
	#print float(avg)/float(len_fil)
	return filtered
			
	


