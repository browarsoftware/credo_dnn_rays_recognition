"""
Author: Tomasz Hachaj, 2020
Department of Signal Processing and Pattern Recognition
Institute of Computer Science in Pedagogical University of Krakow, Poland
https://sppr.up.krakow.pl/hachaj/

Data source:
https://credo.nkg-mn.com/hits.html
"""

import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.applications.nasnet import NASNetLarge
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.densenet import DenseNet201
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet_v2 import MobileNetV2

import keras
from keras.layers import Dense

def enable_tensorflow():
    '''
    Enables tensorflow on GPU
    :return: physical devices reference
    '''
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)
    return physical_devices

def create_model_VGG16(include_top=False):
    '''
    Load VGG16 network pretrained on imagenet
    :param include_top: include classification layer?
    :return: VGG16 network
    '''
    my_model = VGG16(weights='imagenet', include_top=include_top)
    return my_model

def create_model_NASNetLarge(include_top=False):
    '''
    Load NASNetLarge network pretrained on imagenet
    :param include_top: include classification layer?
    :return: NASNetLarge network
    '''
    my_model = NASNetLarge(weights='imagenet',include_top=include_top)
    return my_model

def create_model_MobileNetV2(include_top=False):
    '''
    Load VGG16 network pretrained on imagenet
    :param include_top: include classification layer?
    :return: VGG16 network
    '''
    my_model = MobileNetV2(weights='imagenet', include_top=False)
    return my_model

def create_model_Xception(include_top=False):
    '''
    Load VGG16 network pretrained on imagenet
    :param include_top: include classification layer?
    :return: VGG16 network
    '''
    my_model = Xception(weights='imagenet', include_top=False)
    return my_model

def create_model_DenseNet201(include_top=False):
    '''
    Load VGG16 network pretrained on imagenet
    :param include_top: include classification layer?
    :return: VGG16 network
    '''
    my_model = DenseNet201(weights='imagenet', include_top=False)
    return my_model


def create_dense_nn(input_layer_neurons_count, hidden_layer_neurons_count, output_layer_neurons_count):
    model = keras.Sequential()
    model.add(Dense(hidden_layer_neurons_count, activation="relu", input_dim=input_layer_neurons_count))
    model.add(Dense(output_layer_neurons_count, activation='sigmoid'))
    #model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])
    return model

def create_regression_nn(input_layer_neurons_count, hidden_layer_neurons_count, output_layer_neurons_count):
    model = keras.Sequential()
    model.add(Dense(hidden_layer_neurons_count, activation="relu", input_dim=input_layer_neurons_count))
    model.add(Dense(output_layer_neurons_count, activation='linear'))
    model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])
    return model
