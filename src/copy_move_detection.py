import os
import time
import image_object

def detect_dir(sourceDirectory, outputDirectory, blockSize=32):
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
