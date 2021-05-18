import cv2
import sys

import double_jpeg_compression
import copy_move_cfa
import copy_move_detection
import noise_variance
import os.path as path

from optparse import OptionParser

if __name__ == '__main__':
    # copy-move parameters
    cmd = OptionParser("usage: %prog image_file [options]")
    cmd.add_option('', '--imauto', help='Automatically search identical regions. (default: %default)', default=1)
    cmd.add_option('', '--imblev',help='Blur level for degrading image details. (default: %default)', default=8)
    cmd.add_option('', '--impalred',help='Image palette reduction factor. (default: %default)', default=15)
    cmd.add_option('', '--rgsim', help='Region similarity threshold. (default: %default)', default=5)
    cmd.add_option('', '--rgsize',help='Region size threshold. (default: %default)', default=1.5)
    cmd.add_option('', '--blsim', help='Block similarity threshold. (default: %default)',default=200)
    cmd.add_option('', '--blcoldev', help='Block color deviation threshold. (default: %default)', default=0.2)
    cmd.add_option('', '--blint', help='Block intersection threshold. (default: %default)', default=0.2)
    opt, args = cmd.parse_args()
    if not args:
        cmd.print_help()
        sys.exit()
    im_str = str(args[0])

    input = '..//images//' + im_str
    if not path.exists(input):
        sys.exit("Image not found: {}. Please place the image in the images subdirectory.".format(im_str))

    print('\nRunning double jpeg compression detection...')
    double_compressed = double_jpeg_compression.detect(input)

    if(double_compressed): print('\nDouble compression detected')
    else: print('\nSingle compressed')

    print('\nRunning CFA artifact detection...\n')
    identical_regions_cfa = copy_move_cfa.detect(input, opt, args)
    print('\n' + str(identical_regions_cfa), 'CFA artifacts detected')

    print('\nRunning noise variance inconsistency detection...')
    noise_forgery = noise_variance.detect(input)

    if(noise_forgery): print('\nNoise variance inconsistency detected')
    else: print('\nNo noise variance inconsistency detected')

    print('\nRunning copy-move detection...\n')
    copy_move_detection.detect('../images/', im_str, '../output/', blockSize=32)
    print(identical_regions_cfa, 'identical regions detected')

    if ((not double_compressed) and (identical_regions_cfa == 0) and (not noise_forgery)):
        print('\nNo forgeries were detected - this image has probably not been tampered with.')
    else:
        print('\nSome forgeries were detected - this image may have been tampered with.')
