
import os
import time
import image_object

def detect_dir(sourceDirectory, outputDirectory, blockSize=32):
    """
    Detects all images under a directory
    :param sourceDirectory: directory that contains images to be detected
    :param outputDirectory: output directory
    :param blockSize: the block size of the image pointer (eg. 32, 64, 128)
    The smaller the block size, the more accurate the result is, but takes more time, vice versa.
    :return: None
    """

    timeStamp = time.strftime("%Y%m%d_%H%M%S")  # get current timestamp
    os.makedirs(outputDirectory + timeStamp)    # create a folder named as the current timestamp

    if not os.path.exists(sourceDirectory):
        print("\tError: Source Directory did not exist.")
        return
    elif not os.path.exists(outputDirectory):
        print("\tError: Output Directory did not exist.")
        return

    for fileName in os.listdir(sourceDirectory):
        anImage = image_object.image_object(sourceDirectory, fileName, blockSize, outputDirectory + timeStamp + '/')
        anImage.run()

    print("\tDone.")


def detect(sourceDirectory, fileName, outputDirectory, blockSize=32):
    """
    Detects an image under a specific directory
    :param sourceDirectory: directory that contains images to be detected
    :param fileName: name of the image file to be detected
    :param outputDirectory: output directory
    :param blockSize: the block size of the image pointer (eg. 32, 64, 128)
    The smaller the block size, the more accurate the result is, but takes more time, vice versa.
    :return: None
    """

    if not os.path.exists(sourceDirectory):
        print("\tError: Source Directory did not exist.")
        return
    elif not os.path.exists(sourceDirectory + fileName):
        print("\tError: Image file did not exist.")
        return
    elif not os.path.exists(outputDirectory):
        print("\tError: Output Directory did not exist.")
        return

    singleImage = image_object.image_object(sourceDirectory, fileName, blockSize, outputDirectory)
    imageResultPath = singleImage.run()

    return imageResultPath
