"""
Author: Tomasz Hachaj, 2020
Department of Signal Processing and Pattern Recognition
Institute of Computer Science in Pedagogical University of Krakow, Poland
https://sppr.up.krakow.pl/hachaj/

Data source:
https://credo.nkg-mn.com/hits.html
"""


import cv2
import pandas as pd
import numpy as np
from scipy import ndimage
import os

def center_in_the_center_of_mass(img):
    '''
    Center image in the center of mass
    :param img: input numpy array
    :return: centered numpy array
    '''
    val = ndimage.measurements.center_of_mass(img)
    height, width = img.shape[:2]
    quarter_height, quarter_width = height / 2 - (val[0]), width / 2 - (val[1])
    T = np.float32([[1, 0, quarter_width], [0, 1, quarter_height]])
    img_translation = cv2.warpAffine(img, T, (width, height))
    return img_translation

def process_dataset(path_to_data, path_to_output, path_to_descriptions):
    '''
    Removes bacground from images and centers them in the center of mass
    :param path_to_data: path to folder with data
    :param path_to_output: path to output folder
    :param path_to_descriptions: csv file with file names
    :return:
    '''

    try:
        os.mkdir(path_to_output)
    except OSError:
        a = 0
        a = a + 1
    else:
        a = 0
        a = a + 1

    my_data = pd.read_csv(path_to_descriptions)
    print('Image count: ' + str(my_data.shape[0]))

    #kernel for image morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for a in range(my_data.shape[0]):
        if (a % 1000 == 0):
            print("Processing " + str(a) + " of " + str(my_data.shape[0]))
        #loads data from png file
        img = cv2.imread(path_to_data + str(my_data.iloc[a, 0]) + '.png')
        #converts to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #adaptive tresholding to BW mask
        (thresh, im_bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        #mask dilation (increases size)
        im_bw = cv2.dilate(im_bw, kernel, iterations=1)
        #closing (removes holes)
        im_bw = cv2.morphologyEx(im_bw, cv2.MORPH_CLOSE, kernel)
        #extracts only masked area from grayscale image
        gray[im_bw < 10] = 0
        #center in the center of mass
        #gray = center_in_the_center_of_mass(gray)
        cv2.imwrite(path_to_output + str(my_data.iloc[a, 0]) + '.png', gray)

