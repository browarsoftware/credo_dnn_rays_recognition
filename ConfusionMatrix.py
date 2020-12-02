"""
Author: Tomasz Hachaj, 2020
Department of Signal Processing and Pattern Recognition
Institute of Computer Science in Pedagogical University of Krakow, Poland
https://sppr.up.krakow.pl/hachaj/

Data source:
https://credo.nkg-mn.com/hits.html
"""

import numpy as np
from numpy import genfromtxt
import random
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from numpy import savetxt

def calculate_confusion_matrix(sample_size, my_model_name, path_to_results, path_to_descriptions, my_seed):
    path_help1 = path_to_results + '/' + my_model_name + '.bin.npy'
    path_help2 = path_to_results + '/' + my_model_name + '_2.bin.npy'

    result = np.load(path_help1)
    result_pred = np.load(path_help2)

    Y = genfromtxt(path_to_descriptions, delimiter=',', skip_header=1)
    Y = Y[:, [1, 2, 3, 4]]
    random.seed(my_seed)
    my_random_sample = random.sample(range(Y.shape[0]), sample_size)
    Y = Y[my_random_sample, ]

    y_true = np.zeros(Y.shape[0])
    for a in range(Y.shape[0]):
        acc = Y[a,]
        my_prediction_sort = np.argsort(acc)[::-1]
        y_true[a] = my_prediction_sort[0]

    print(y_true.shape)

    y_pred = result[:,0]
    cm = confusion_matrix(y_true, y_pred)
    #print(pp)
    cmn = confusion_matrix(y_true, y_pred, normalize='true')
    #print(pp)

    # Using 'auto'/'sum_over_batch_size' reduction type.
    bce = tf.keras.losses.BinaryCrossentropy()
    #print(bce(Y, result_pred).numpy())
    mybce = bce(Y, result_pred).numpy()

    mse = tf.keras.losses.MeanSquaredError()
    mseres = mse(Y, result_pred).numpy()

    np.save(path_to_results + '/' + my_model_name + '_confusion_matrix.bin', cm)
    np.save(path_to_results + '/' + my_model_name + '_confusion_matrix_normalize.bin', cmn)
    np.save(path_to_results + '/' + my_model_name + '_mybce.bin', mybce)

    savetxt(path_to_results + '/' + my_model_name + '_confusion_matrix.csv', cm, delimiter=',')
    savetxt(path_to_results + '/' + my_model_name + '_confusion_matrix_normalize.csv', cmn, delimiter=',')
    savetxt(path_to_results + '/' + my_model_name + '_mybce.csv', np.asarray([mybce]), delimiter=',')

    savetxt(path_to_results + '/' + my_model_name + '_mymse.csv', np.asarray([mseres]), delimiter=',')
