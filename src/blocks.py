import numpy as np
from sklearn.decomposition import PCA

class blocks(object):
    def __init__(self, grayscaleImageBlock, rgbImageBlock, x, y, blockDimension):
        self.imageGrayscale = grayscaleImageBlock  # block of grayscale image
        self.imageGrayscalePixels = self.imageGrayscale.load()

        if rgbImageBlock is not None:
            self.imageRGB = rgbImageBlock
            self.imageRGBPixels = self.imageRGB.load()
            self.isImageRGB = True
        else:
            self.isImageRGB = False

        self.coor = (x, y)
        self.blockDimension = blockDimension

    def computeBlock(self):
        blockDataList = []
        blockDataList.append(self.coor)
        blockDataList.append(self.computeCharaFeatures(4))
        blockDataList.append(self.computePCA(6))
        return blockDataList

    def computePCA(self, precision):
        PCAModule = PCA(n_components=1)
        if self.isImageRGB:
            imageArray = np.array(self.imageRGB)
            r = imageArray[:, :, 0]
            g = imageArray[:, :, 1]
            b = imageArray[:, :, 2]

            concatenatedArray = np.concatenate((r, np.concatenate((g, b), axis=0)), axis=0)
            PCAModule.fit_transform(concatenatedArray)
            principalComponents = PCAModule.components_
            preciseResult = [round(element, precision) for element in list(principalComponents.flatten())]
            return preciseResult
        else:
            imageArray = np.array(self.imageGrayscale)
            PCAModule.fit_transform(imageArray)
            principalComponents = PCAModule.components_
            preciseResult = [round(element, precision) for element in list(principalComponents.flatten())]
            return preciseResult

    def computeCharaFeatures(self, precision):
        characteristicFeaturesList = []

        c4_part1 = 0
        c4_part2 = 0
        c5_part1 = 0
        c5_part2 = 0
        c6_part1 = 0
        c6_part2 = 0
        c7_part1 = 0
        c7_part2 = 0

        if self.isImageRGB:
            sumOfRedPixelValue = 0
            sumOfGreenPixelValue = 0
            sumOfBluePixelValue = 0
            for yCoordinate in range(0, self.blockDimension):  # compute sum of the pixel value
                for xCoordinate in range(0, self.blockDimension):
                    tmpR, tmpG, tmpB = self.imageRGBPixels[xCoordinate, yCoordinate]
                    sumOfRedPixelValue += tmpR
                    sumOfGreenPixelValue += tmpG
                    sumOfBluePixelValue += tmpB

            sumOfPixels = self.blockDimension * self.blockDimension
            sumOfRedPixelValue = sumOfRedPixelValue / (sumOfPixels)  # mean from each of the colorspaces
            sumOfGreenPixelValue = sumOfGreenPixelValue / (sumOfPixels)
            sumOfBluePixelValue = sumOfBluePixelValue / (sumOfPixels)

            characteristicFeaturesList.append(sumOfRedPixelValue)
            characteristicFeaturesList.append(sumOfGreenPixelValue)
            characteristicFeaturesList.append(sumOfBluePixelValue)

        else:
            characteristicFeaturesList.append(0)
            characteristicFeaturesList.append(0)
            characteristicFeaturesList.append(0)

        for yCoordinate in range(0, self.blockDimension):
            for xCoordinate in range(0, self.blockDimension):
                if yCoordinate <= self.blockDimension / 2:
                    c4_part1 += self.imageGrayscalePixels[xCoordinate, yCoordinate]
                else:
                    c4_part2 += self.imageGrayscalePixels[xCoordinate, yCoordinate]
                if xCoordinate <= self.blockDimension / 2:
                    c5_part1 += self.imageGrayscalePixels[xCoordinate, yCoordinate]
                else:
                    c5_part2 += self.imageGrayscalePixels[xCoordinate, yCoordinate]
                if xCoordinate - yCoordinate >= 0:
                    c6_part1 += self.imageGrayscalePixels[xCoordinate, yCoordinate]
                else:
                    c6_part2 += self.imageGrayscalePixels[xCoordinate, yCoordinate]
                if xCoordinate + yCoordinate <= self.blockDimension:
                    c7_part1 += self.imageGrayscalePixels[xCoordinate, yCoordinate]
                else:
                    c7_part2 += self.imageGrayscalePixels[xCoordinate, yCoordinate]

        characteristicFeaturesList.append(float(c4_part1) / float(c4_part1 + c4_part2))
        characteristicFeaturesList.append(float(c5_part1) / float(c5_part1 + c5_part2))
        characteristicFeaturesList.append(float(c6_part1) / float(c6_part1 + c6_part2))
        characteristicFeaturesList.append(float(c7_part1) / float(c7_part1 + c7_part2))

        preciseResult = [round(element, precision) for element in characteristicFeaturesList]
        return preciseResult
