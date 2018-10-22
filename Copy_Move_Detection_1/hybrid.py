from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import math
import scipy as sp
import zernikemoments as zmm
import matching_zernike
from PIL import Image,ImageDraw
import sys
import zernike as zn
import filtering_blocks as fb
INF = 99999999

def fun(path,number):

    #Load the image and apply SLIC
    image = cv2.imread(path)

    segments = slic(img_as_float(image), n_segments = 100, sigma = 5)
    #print type(segments)


    #Counting the number of superpixels
    number_of_segments = 0
    for (i, segVal) in enumerate(np.unique(segments)):
        number_of_segments += 1


    #Counting number of rows by columns in segments 2D list
    rows = segments.shape[0]
    columns = segments.shape[1]
    #print "rows=%d columns=%d" % (rows,columns)


    #Applying SIFT
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    (kps, descs) = sift.detectAndCompute(gray, None)
    kps_total = len(kps)


    #Num_pixles stores the number of pixels in each superpixel
    num_pixels = [0] * number_of_segments
    for x in range(0,rows) :
        for y in range(0,columns):
            temp = segments[x,y]
            num_pixels[temp] += 1


    #Num_keypoints counts the number of keypoints in each superpixel
    num_keypoints = [0] * number_of_segments
    for i in range(0,kps_total) :
        (p,q) = kps[i].pt
        p = int(round(p))
        q = int(round(q))
        t = segments[q,p]
        num_keypoints[t] += 1


    #smooth and non-smooth
    ratio = [0] * number_of_segments
    #threshold for smooth & non-smooth
    T = 0.009
    is_smooth = [0] * number_of_segments
    for (i, segVal) in enumerate(np.unique(segments)):
        ratio[i] = float(num_keypoints[i])/float(num_pixels[i])
        if(ratio[i] < T):
            is_smooth[i] = 1




    #Finding all the indices in the keypoint regions
    non_smooth_kps = []
    kp_vs_segmentNum = []
    num_non_smooth_kps = 0
    for i in range(0,kps_total):
        (p,q) = kps[i].pt
        p = int(round(p))
        q = int(round(q))
        t = segments[q,p]             # t stores segment value in which i'th keypoint exists
        kp_vs_segmentNum.append(t)
        if(is_smooth[t] == 0):
            non_smooth_kps.append(i)
        num_non_smooth_kps += 1     #stores no. of keypoint in non smooth segments


    #To create desc_2 storing only non smooth regions' keypoints' descriptors
    descs_2 = np.zeros(shape=(num_non_smooth_kps,128))
    for i in range(0,len(non_smooth_kps)) :
        p = non_smooth_kps[i]
        descs_2[i] = descs[p]



    #FLANN Based Matcher
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=100)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(descs,descs,k=3)

    print("No. of keypoints after FLANN matching: %d" % (len(matches)))

    # To remove the very near matches
    filtered_output=[]  # it will store descriptors's indices of matched pairs - (i,j)
    for i in range(0,len(matches)):
        for j in range(1,len(matches[i])):
            ind1 = matches[i][j].queryIdx
            ind2 = matches[i][j].trainIdx
            (p1,q1) = kps[ind1].pt
            p1 = int(round(p1))
            q1 = int(round(q1))
            t1 = segments[q1,p1]
            (p2,q2) = kps[ind2].pt
            p2 = int(round(p2))
            q2 = int(round(q2))
            t2 = segments[q2,p2]
            if(is_smooth[t1]==0 and is_smooth[t2]==0):
                dis_between = matches[i][j].distance
                if(dis_between<150):
                    filtered_output.append([ind1,ind2])

    # draw matched pairs in image:
    im = Image.open(path);
    draw = ImageDraw.Draw(im)

    #showing matches after filtering
    for i in range(0,len(filtered_output)):
        ind1=filtered_output[i][0]
        ind2=filtered_output[i][1]
        (p1,q1) = kps[ind1].pt
        p1 = int(round(p1))
        q1 = int(round(q1))
        (p2,q2) = kps[ind2].pt
        p2 = int(round(p2))
        q2 = int(round(q2))
        draw.line((p1,q1,p2,q2), fill="yellow", width=0)


    print("No. of keypoints after filtering: %d" % (len(filtered_output)))

    im2 = cv2.imread(path)
    cv2.rectangle(im2,(50,50),(150,150),(255,0,0,),-1)

    name = str(number)

    if(len(filtered_output) >=14 ): #experimental value
        im.save(path+"_res.png")
        return

    #------------------------------------carry on with block matching procedure using Zernike Moments--------------------------------------------


    else:
    #Counting number of rows by columns in segments 2D list
        width = segments.shape[1]     # ----------this is basically number of columns
        height = segments.shape[0]    # ----------this is basically number of rows

        #Applying SIFT
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        (kps, descs) = sift.detectAndCompute(gray, None)
        kps_total = len(kps)


        #Num_pixles stores the number of pixels in each superpixel
        num_pixels = [0] * number_of_segments
        for i in range(0,height) :
            for j in range(0,width) :
                temp = segments[i,j]
                num_pixels[temp] += 1


        im = Image.open(path);       # method to draw matches on image
        draw = ImageDraw.Draw(im)


        #Num_keypoints counts the number of keypoints in each superpixel
        num_keypoints = [0] * number_of_segments
        for i in range(0,kps_total) :
            (p,q) = kps[i].pt
            p = int(round(p))
            q = int(round(q))
            t = segments[q,p]
            num_keypoints[t] += 1


        dictionary = {}
        #smooth and non-smooth
        ratio = [0] * number_of_segments
        T = 0.005
        is_smooth = [0] * number_of_segments
        num_smooth_regions = 0
        for (i, segVal) in enumerate(np.unique(segments)):
            ratio[i] = float(num_keypoints[i])/float(num_pixels[i])
            if(ratio[i] < T) :
                is_smooth[i] = 1
                num_smooth_regions += 1
                dictionary[i] = [INF,INF,-INF,-INF]  # xmin,ymin,xmax,ymax


        #Finding all the indices in the keypoint regions
        non_smooth_kps = []
        num_non_smooth_kps = 0
        for i in range(0,kps_total) :
            (p,q) = kps[i].pt
            p = int(round(p))
            q = int(round(q))
            t = segments[q,p]          # t stores segment value in which i'th keypoint exists
            if(is_smooth[t] == 0) :
                non_smooth_kps.append(i)  # keypoints which are in non smooth regions
                num_non_smooth_kps += 1



        #-----------------------dictionary stores the xmin,ymin, xmax and ymax values of each non-smooth block---------------------------

        im = Image.open(path);       # method to draw on image
        draw = ImageDraw.Draw(im)
        for i in range(0,height) :
            for j in range(0,width) :
                temp = segments[i,j]      #fetches segment value of pixel (x,y) in segment 2d numpy array
                if(is_smooth[temp] == 1):
                    if(dictionary[temp][0] > i):
                        dictionary[temp][0] = i
                    if(dictionary[temp][1] > j):
                        dictionary[temp][1] = j
                    if(dictionary[temp][2] < i):
                        dictionary[temp][2] = i
                    if(dictionary[temp][3] < j):
                        dictionary[temp][3] = j


        zernike_values = []   # storing all zernike values
        indices = []  # to store indices for smooth segments



    #-----Dividing into overlapping blocks and getting each OB's features for matching using the sernike moments function in mahotas->opencv-----
        for k in dictionary:
            xmin = dictionary[k][0]
            ymin = dictionary[k][1]
            xmax = dictionary[k][2]
            ymax = dictionary[k][3]

            # to draw the non smooth regions

            for r in range(xmin, xmax - 16, 6):
                    for c in range(ymin, ymax - 16, 6):
                        window = image[r:r+16,c:c+16]  #one overlapping block
                        mom_val=[]
                        mom_val=(zn.cal_zernike(window)).tolist()  # to convert ndarray into list
                        zernike_values.append(mom_val)
                        indices.append([r+8,c+8])   # storing boundary of overlapping window


        #----------------------------Block matching by lexicographic sorting and euclidean distance matching:-----------------------------

        zernike_values, indices = zip(*sorted(zip(zernike_values, indices)))   #  doing the sorting here----------------------------

        matching_pairs = []
        matching_pairs = matching_zernike.check(zernike_values,indices)
        #returns matched pairs of zernike values and we will find segmnt num. for that later


        print("No. of regions block matching: %d" % (len(matching_pairs)))

        filtered=[]
        filtered=fb.filter_outliers(matching_pairs,indices,zernike_values)

        print("No. of regions after filtered in block regions: %d" % (len(filtered)))


        # showing matches after filtering
        #im = Image.open(path);
        #draw = ImageDraw.Draw(im)
        for i in range(0,len(filtered)):
            cy1,cx1,cy2,cx2 = filtered[i]
            cv2.rectangle(image,(cx1-8,cy1-8),(cx1+8,cy1+8),(255,0,0),-1)
            cv2.rectangle(image,(cx2-8,cy2-8),(cx2+8,cy2+8),(255,0,0,),-1)
            img = Image.fromarray(image, 'RGB')
        #im.show()
        if(len(filtered)==0 or len(matching_pairs)==0):
            return

        img.save(name+"_res.png")
