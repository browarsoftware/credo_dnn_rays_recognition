"""
Author: Tomasz Hachaj, 2020
Department of Signal Processing and Pattern Recognition
Institute of Computer Science in Pedagogical University of Krakow, Poland
https://sppr.up.krakow.pl/hachaj/

Data source:
https://credo.nkg-mn.com/hits.html
"""

from numpy import savetxt
from tensorflow_utils import create_dense_nn

def make_predictions(sample_size, hidden_layer_neurons_count, output_layer_neurons_count, my_model_name,
                     output_features_file, path_to_descriptions, path_to_checkpoints, path_to_results, my_seed):

    try:
        import os
        os.mkdir(path_to_results + '/')
    except OSError:
        a = 0
        a = a + 1
    else:
        a = 0
        a = a + 1

    import glob
    import os
    print('Opening directory ' + path_to_checkpoints + "/" + my_model_name + "/")
    list_of_files = glob.glob(path_to_checkpoints + "/" + my_model_name + "/" + '*.hdf5') # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    import numpy as np
    from numpy import genfromtxt
    Y = genfromtxt(path_to_descriptions, delimiter=',', skip_header=1)
    Y = Y[:, [1, 2, 3, 4]]


    X = genfromtxt(output_features_file, delimiter=',')
    input_layer_neurons_count = X.shape[1]
    model = create_dense_nn(input_layer_neurons_count, hidden_layer_neurons_count, output_layer_neurons_count)
    model.load_weights(latest_file)

    import random
    random.seed(my_seed)
    my_random_sample = random.sample(range(X.shape[0]), sample_size)


    mask = np.ones(X.shape[0], dtype=bool)
    mask[my_random_sample] = False

    how_many_to_check = Y.shape[1]
    #VALID
    X = X[my_random_sample, ]
    Y = Y[my_random_sample, ]
    how_much_data = X.shape[0]

    res_array = np.zeros([Y.shape[0], how_many_to_check])
    res_array2 = np.zeros([Y.shape[0], how_many_to_check])

    ################################
    for my_id in range(Y.shape[0]):
        if my_id % 100 == 0:
            print(str(my_id) + " of " + str(Y.shape[0]))

        my_x = X[my_id, :]
        my_x = my_x.reshape(1, my_x.shape[0])
        my_prediction = model.predict(my_x)
        my_prediction = my_prediction[0]

        res_array2[my_id,] = my_prediction

        my_prediction_sort = np.argsort(my_prediction)[::-1]
        for a in range(how_many_to_check):
            aa = my_prediction_sort[0:how_many_to_check]
            res_array[my_id, a] = aa[a]
        print(res_array[my_id,:])

    np.save(path_to_results + '/' + my_model_name + '.bin', res_array)
    np.save(path_to_results + '/' + my_model_name + '_2.bin', res_array2)
    savetxt(path_to_results + '/' + my_model_name + '.csv', res_array, delimiter=',')
    savetxt(path_to_results + '/' + my_model_name + '_2.csv', res_array2, delimiter=',')

