"""
Author: Tomasz Hachaj, 2020
Department of Signal Processing and Pattern Recognition
Institute of Computer Science in Pedagogical University of Krakow, Poland
https://sppr.up.krakow.pl/hachaj/

Data source:
https://credo.nkg-mn.com/hits.html
"""

#import pandas as pd
import numpy as np
import os
from tensorflow_utils import create_dense_nn, create_regression_nn

def transfer_learning(sample_size, my_seed, batch_size, epochs, initial_learning_rate, learning_rate_step,
                       path_to_checkpoints, DataFile, path_to_features, path_to_descriptions,
                       hidden_layer_neurons_count, output_layer_neurons_count):
    try:
        os.mkdir(path_to_checkpoints + '/')
    except OSError:
        a = 0
        a = a + 1
    else:
        a = 0
        a = a + 1

    try:
        os.mkdir(path_to_checkpoints + '/' + DataFile + '/')
    except OSError:
        a = 0
        a = a + 1
    else:
        a = 0
        a = a + 1

    from numpy import genfromtxt
    X = genfromtxt(path_to_features, delimiter=',')

    print(X.shape)

    input_layer_neurons_count = X.shape[1]

    model = create_dense_nn(input_layer_neurons_count, hidden_layer_neurons_count, output_layer_neurons_count)
    #model = create_regression_nn(input_layer_neurons_count, hidden_layer_neurons_count, output_layer_neurons_count)


    Y = genfromtxt(path_to_descriptions, delimiter=',', skip_header=1)
    Y = Y[:, [1, 2, 3, 4]]

    import random
    random.seed(my_seed)
    my_random_sample = random.sample(range(X.shape[0]), sample_size)

    mask = np.ones(X.shape[0], dtype=bool)
    mask[my_random_sample] = False

    print("X = " + str(X.shape))
    print("Y = " + str(Y.shape))

    #TRAIN
    X = X[mask, ]
    Y = Y[mask, ]

    step_size_train=X.shape[0]//batch_size
    print(X.shape)
    print(Y.shape)

    from keras.callbacks import ModelCheckpoint
    from keras.callbacks import LearningRateScheduler
    from keras.callbacks import CSVLogger
    # checkpoint

    csv_logger = CSVLogger(path_to_checkpoints + "/" + DataFile + '.log')
    filepath= path_to_checkpoints + '/' + DataFile + '/' + DataFile + "-{epoch:02d}-{accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max', save_weights_only=True)

    def lr_scheduler(epoch, lr):
        #if epoch == 1:
        #    lr = 0.01
        if epoch % learning_rate_step == 0 and epoch > 0:
            lr = lr * initial_learning_rate
        return lr

    callbacks_list = [checkpoint,LearningRateScheduler(lr_scheduler, verbose=1),csv_logger]

    model.fit(x = X,
              y = Y,
              shuffle=True,
              batch_size = batch_size,
              steps_per_epoch=step_size_train,
              epochs=epochs,
              callbacks=callbacks_list)
'''
sample_size = 200
batch_size = 64
epochs=4000
initial_learning_rate = 0.1
learning_rate_step = 2000
path_to_checkpoints = "checkpointsTest/VGG16/"
DataFile = 'VGG16'
path_to_features = 'Features\VGG16Features3.txt'
path_to_descriptions = 'd:\\dane\\credo\\dane2.txt'
my_seed = 4321
hidden_layer_neurons_count = 128
output_layer_neurons_count = 4

transfer_learining(sample_size, my_seed, batch_size, epochs, initial_learning_rate, learning_rate_step,
                       path_to_checkpoints, DataFile, path_to_features, path_to_descriptions,
                       hidden_layer_neurons_count, output_layer_neurons_count)
'''