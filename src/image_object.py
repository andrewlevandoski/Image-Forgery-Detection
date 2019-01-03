from PIL import Image
from math import pow
from tqdm import tqdm, trange

import scipy.misc
import numpy as np
import builtins
import time

import container
import blocks

class image_object(object):
    def __init__(self, imageDirectory, imageName, blockDimension, outputDirectory):
        print("\tStep 1 of 4: Object and variable initialization")

        # image parameter
        self.imageOutputDirectory = outputDirectory
        self.imagePath = imageName
        self.imageData = Image.open(imageDirectory + imageName)
        self.imageWidth, self.imageHeight = self.imageData.size  # height = vertikal, width = horizontal

        if self.imageData.mode != 'L':  # L means grayscale
            self.isThisRGBImage = True
            self.imageData = self.imageData.convert('RGB')
            RGBImagePixels = self.imageData.load()
            self.imageGrayscale = self.imageData.convert(
                'L')  # creates a grayscale version of current image to be used later
            GrayscaleImagePixels = self.imageGrayscale.load()

            for yCoordinate in range(0, self.imageHeight):
                for xCoordinate in range(0, self.imageWidth):
                    redPixelValue, greenPixelValue, bluePixelValue = RGBImagePixels[xCoordinate, yCoordinate]
                    GrayscaleImagePixels[xCoordinate, yCoordinate] = int(0.299 * redPixelValue) + int(
                        0.587 * greenPixelValue) + int(0.114 * bluePixelValue)
        else:
            self.isThisRGBImage = False
            self.imageData = self.imageData.convert('L')

        # algorithm's parameters from the first paper
        self.N = self.imageWidth * self.imageHeight
        self.blockDimension = blockDimension
        self.b = self.blockDimension * self.blockDimension
        self.Nb = (self.imageWidth - self.blockDimension + 1) * (self.imageHeight - self.blockDimension + 1)
        self.Nn = 2  # amount of neighboring block to be evaluated
        self.Nf = 188  # minimum treshold of the offset's frequency
        self.Nd = 50  # minimum treshold of the offset's magnitude

        # algorithm's parameters from the second paper
        self.P = (1.80, 1.80, 1.80, 0.0125, 0.0125, 0.0125, 0.0125)
        self.t1 = 2.80
        self.t2 = 0.02

        print('\t', self.Nb, self.isThisRGBImage)

        # container initialization to later contains several data
        self.featurescontainer = container.container()
        self.blockPaircontainer = container.container()
        self.offsetDictionary = {}

    def run(self):
        # time logging (optional, for evaluation purpose)
        self.compute()
        startTimestamp = time.time()
        timestampAfterComputing = time.time()
        self.sort()
        timestampAfterSorting = time.time()
        self.analyze()
        timestampAfterAnalyze = time.time()
        imageResultPath = self.reconstruct()
        timestampAfterImageCreation = time.time()

        print("\tComputing time :", timestampAfterComputing - startTimestamp, "seconds")
        print("\tSorting time   :", timestampAfterSorting - timestampAfterComputing, "seconds")
        print("\tAnalyzing time :", timestampAfterAnalyze - timestampAfterSorting, "seconds")
        print("\tImage creation :", timestampAfterImageCreation - timestampAfterAnalyze, "seconds")

        totalRunningTimeInSecond = timestampAfterImageCreation - startTimestamp
        totalMinute, totalSecond = divmod(totalRunningTimeInSecond, 60)
        totalHour, totalMinute = divmod(totalMinute, 60)
        print("\tTotal time    : %d:%02d:%02d second" % (totalHour, totalMinute, totalSecond), '\n')
        return imageResultPath

    def compute(self):
        print("\tStep 2 of 4: Computing characteristic features")

        imageWidthOverlap = self.imageWidth - self.blockDimension
        imageHeightOverlap = self.imageHeight - self.blockDimension

        if self.isThisRGBImage:
            for i in tqdm(range(0, imageWidthOverlap + 1, 1)):
                for j in range(0, imageHeightOverlap + 1, 1):
                    imageBlockRGB = self.imageData.crop((i, j, i + self.blockDimension, j + self.blockDimension))
                    imageBlockGrayscale = self.imageGrayscale.crop(
                        (i, j, i + self.blockDimension, j + self.blockDimension))
                    imageBlock = blocks.blocks(imageBlockGrayscale, imageBlockRGB, i, j, self.blockDimension)
                    self.featurescontainer.addBlock(imageBlock.computeBlock())
        else:
            for i in range(imageWidthOverlap + 1):
                for j in range(imageHeightOverlap + 1):
                    imageBlockGrayscale = self.imageData.crop((i, j, i + self.blockDimension, j + self.blockDimension))
                    imageBlock = blocks.blocks(imageBlockGrayscale, None, i, j, self.blockDimension)
                    self.featurescontainer.addBlock(imageBlock.computeBlock())

    def sort(self):
        self.featurescontainer.sortFeatures()

    def analyze(self):
        print("\tStep 3 of 4:Pairing image blocks")
        z = 0
        time.sleep(0.1)
        featurecontainerLength = self.featurescontainer.getLength()
        for i in tqdm(range(featurecontainerLength)):
            for j in range(i + 1, featurecontainerLength):
                result = self.isValid(i, j)
                if result[0]:
                    self.addDict(self.featurescontainer.container[i][0], self.featurescontainer.container[j][0], result[1])
                    z += 1
                else:
                    break

    def isValid(self, firstBlock, secondBlock):
        if abs(firstBlock - secondBlock) < self.Nn:
            iFeature = self.featurescontainer.container[firstBlock][1]
            jFeature = self.featurescontainer.container[secondBlock][1]

            # check the validity of characteristic features according to the second paper
            if abs(iFeature[0] - jFeature[0]) < self.P[0]:
                if abs(iFeature[1] - jFeature[1]) < self.P[1]:
                    if abs(iFeature[2] - jFeature[2]) < self.P[2]:
                        if abs(iFeature[3] - jFeature[3]) < self.P[3]:
                            if abs(iFeature[4] - jFeature[4]) < self.P[4]:
                                if abs(iFeature[5] - jFeature[5]) < self.P[5]:
                                    if abs(iFeature[6] - jFeature[6]) < self.P[6]:
                                        if abs(iFeature[0] - jFeature[0]) + abs(iFeature[1] - jFeature[1]) + abs(
                                                        iFeature[2] - jFeature[2]) < self.t1:
                                            if abs(iFeature[3] - jFeature[3]) + abs(iFeature[4] - jFeature[4]) + abs(
                                                            iFeature[5] - jFeature[5]) + abs(
                                                        iFeature[6] - jFeature[6]) < self.t2:

                                                # compute the pair's offset
                                                iCoordinate = self.featurescontainer.container[firstBlock][0]
                                                jCoordinate = self.featurescontainer.container[secondBlock][0]

                                                # Non Absolute Robust Detection Method
                                                offset = (
                                                    iCoordinate[0] - jCoordinate[0], iCoordinate[1] - jCoordinate[1])

                                                # compute the pair's magnitude
                                                magnitude = np.sqrt(pow(offset[0], 2) + pow(offset[1], 2))
                                                if magnitude >= self.Nd:
                                                    return 1, offset
        return 0,

    def addDict(self, firstCoordinate, secondCoordinate, pairOffset):
        if pairOffset in self.offsetDictionary:
            self.offsetDictionary[pairOffset].append(firstCoordinate)
            self.offsetDictionary[pairOffset].append(secondCoordinate)
        else:
            self.offsetDictionary[pairOffset] = [firstCoordinate, secondCoordinate]

    def reconstruct(self):
        print("\tStep 4 of 4: Image reconstruction")

        # create an array as the canvas of the final image
        groundtruthImage = np.zeros((self.imageHeight, self.imageWidth))
        linedImage = np.array(self.imageData.convert('RGB'))

        for key in sorted(self.offsetDictionary, key=lambda key: builtins.len(self.offsetDictionary[key]),
                          reverse=True):
            if self.offsetDictionary[key].__len__() < self.Nf * 2:
                break
            print('\t', key, self.offsetDictionary[key].__len__())
            for i in range(self.offsetDictionary[key].__len__()):
                # The original image (grayscale)
                for j in range(self.offsetDictionary[key][i][1],
                               self.offsetDictionary[key][i][1] + self.blockDimension):
                    for k in range(self.offsetDictionary[key][i][0],
                                   self.offsetDictionary[key][i][0] + self.blockDimension):
                        groundtruthImage[j][k] = 255

        # creating a line edge from the original image (for the visual purpose)
        for xCoordinate in range(2, self.imageHeight - 2):
            for yCordinate in range(2, self.imageWidth - 2):
                if groundtruthImage[xCoordinate, yCordinate] == 255 and \
                        (groundtruthImage[xCoordinate + 1, yCordinate] == 0 or groundtruthImage[
                                xCoordinate - 1, yCordinate] == 0 or
                                 groundtruthImage[xCoordinate, yCordinate + 1] == 0 or groundtruthImage[
                            xCoordinate, yCordinate - 1] == 0 or
                                 groundtruthImage[xCoordinate - 1, yCordinate + 1] == 0 or groundtruthImage[
                                xCoordinate + 1, yCordinate + 1] == 0 or
                                 groundtruthImage[xCoordinate - 1, yCordinate - 1] == 0 or groundtruthImage[
                                xCoordinate + 1, yCordinate - 1] == 0):

                    # creating the edge line, respectively left-upper, right-upper, left-down, right-down
                    if groundtruthImage[xCoordinate - 1, yCordinate] == 0 and \
                                    groundtruthImage[xCoordinate, yCordinate - 1] == 0 and \
                                    groundtruthImage[xCoordinate - 1, yCordinate - 1] == 0:
                        linedImage[xCoordinate - 2:xCoordinate, yCordinate, 1] = 255
                        linedImage[xCoordinate, yCordinate - 2:yCordinate, 1] = 255
                        linedImage[xCoordinate - 2:xCoordinate, yCordinate - 2:yCordinate, 1] = 255
                    elif groundtruthImage[xCoordinate + 1, yCordinate] == 0 and \
                                    groundtruthImage[xCoordinate, yCordinate - 1] == 0 and \
                                    groundtruthImage[xCoordinate + 1, yCordinate - 1] == 0:
                        linedImage[xCoordinate + 1:xCoordinate + 3, yCordinate, 1] = 255
                        linedImage[xCoordinate, yCordinate - 2:yCordinate, 1] = 255
                        linedImage[xCoordinate + 1:xCoordinate + 3, yCordinate - 2:yCordinate, 1] = 255
                    elif groundtruthImage[xCoordinate - 1, yCordinate] == 0 and \
                                    groundtruthImage[xCoordinate, yCordinate + 1] == 0 and \
                                    groundtruthImage[xCoordinate - 1, yCordinate + 1] == 0:
                        linedImage[xCoordinate - 2:xCoordinate, yCordinate, 1] = 255
                        linedImage[xCoordinate, yCordinate + 1:yCordinate + 3, 1] = 255
                        linedImage[xCoordinate - 2:xCoordinate, yCordinate + 1:yCordinate + 3, 1] = 255
                    elif groundtruthImage[xCoordinate + 1, yCordinate] == 0 and \
                                    groundtruthImage[xCoordinate, yCordinate + 1] == 0 and \
                                    groundtruthImage[xCoordinate + 1, yCordinate + 1] == 0:
                        linedImage[xCoordinate + 1:xCoordinate + 3, yCordinate, 1] = 255
                        linedImage[xCoordinate, yCordinate + 1:yCordinate + 3, 1] = 255
                        linedImage[xCoordinate + 1:xCoordinate + 3, yCordinate + 1:yCordinate + 3, 1] = 255

                    # creating the straigh line, respectively upper, down, left, right line
                    elif groundtruthImage[xCoordinate, yCordinate + 1] == 0:
                        linedImage[xCoordinate, yCordinate + 1:yCordinate + 3, 1] = 255
                    elif groundtruthImage[xCoordinate, yCordinate - 1] == 0:
                        linedImage[xCoordinate, yCordinate - 2:yCordinate, 1] = 255
                    elif groundtruthImage[xCoordinate - 1, yCordinate] == 0:
                        linedImage[xCoordinate - 2:xCoordinate, yCordinate, 1] = 255
                    elif groundtruthImage[xCoordinate + 1, yCordinate] == 0:
                        linedImage[xCoordinate + 1:xCoordinate + 3, yCordinate, 1] = 255

        timeStamp = time.strftime("%Y%m%d_%H%M%S")
        scipy.misc.imsave(self.imageOutputDirectory + timeStamp + "_" + self.imagePath, groundtruthImage)
        scipy.misc.imsave(self.imageOutputDirectory + timeStamp + "_lined_" + self.imagePath, linedImage)

        return self.imageOutputDirectory + timeStamp + "_lined_" + self.imagePath
