"""
Author: Tomasz Hachaj, 2020
Department of Signal Processing and Pattern Recognition
Institute of Computer Science in Pedagogical University of Krakow, Poland
https://sppr.up.krakow.pl/hachaj/

Data source:
https://credo.nkg-mn.com/hits.html
"""

import pandas as pd
import numpy as np
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing import image
from keras.models import Model
from keras.applications.nasnet import preprocess_input
from PIL import ImageFile
import os
import tensorflow as tf
#enable loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


def generate_features(my_model, my_model_name, path_to_descriptions, path_to_data, output_file, output_features_dir):

    try:
        os.mkdir(output_features_dir)
    except OSError:
        a = 0
        a = a + 1
    else:
        a = 0
        a = a + 1

    csv_ok = pd.read_csv(path_to_descriptions)
    x = my_model.output
    x = GlobalAveragePooling2D()(x)

    model = Model(inputs=my_model.input, outputs=x)

    for a in range(csv_ok.shape[0]):
        if a % 100 == 0:
            print(my_model_name + " " + str(a) + " of " + str(csv_ok.shape[0]))
        my_file = str(csv_ok.iloc[a, 0])
        img_path = path_to_data + my_file + '.png'
        img = image.load_img(img_path, target_size=(60, 60))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)

        file_object = open(output_file, 'a')
        features = features[0]
        for b in range(features.shape[0]):
            if b > 0:
                file_object.write(",")
            file_object.write(str(features[b]))
        file_object.write('\n')
        file_object.close()
