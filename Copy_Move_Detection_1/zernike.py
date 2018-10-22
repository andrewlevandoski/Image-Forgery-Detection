import zernikemoments as zmm
import numpy as np
import argparse
import _pickle as cPickle
import glob
import cv2
from PIL import Image

def cal_zernike(image):
	desc = zmm.ZernikeMoments(21)
	index = {}
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	moments2 = desc.describe(image)

	return moments2
