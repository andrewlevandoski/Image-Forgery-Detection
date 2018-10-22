from PIL import Image, ImageDraw

import numpy as np
import argparse
import _pickle as cPickle
import glob
import cv2
import os
import hybrid

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--sprites", required = True, help = "Path where the sprites will be stored")
args = vars(ap.parse_args())

# loop over the sprite images
i = 10
for spritePath in glob.glob(args["sprites"] + "/*.png"):
	print(spritePath)
	image = cv2.imread(spritePath)
	hybrid.fun(spritePath,i)
	i = i+1

print('done')
