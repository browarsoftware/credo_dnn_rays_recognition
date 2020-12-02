"""
Author: Tomasz Hachaj, 2020
Department of Signal Processing and Pattern Recognition
Institute of Computer Science in Pedagogical University of Krakow, Poland
https://sppr.up.krakow.pl/hachaj/

Data source:
https://credo.nkg-mn.com/hits.html
"""

from prepare_img import process_dataset
from tensorflow_utils import enable_tensorflow, create_model_VGG16, create_model_NASNetLarge, \
    create_model_Xception, create_model_DenseNet201, create_model_MobileNetV2
from generate_features import generate_features
from transferlearning import transfer_learning
from Predictions import make_predictions
from ConfusionMatrix import calculate_confusion_matrix

#Configurations
path_to_data = 'data\\png\\'
path_to_output = 'data\\png_processed\\'
path_to_descriptions = 'data\\data.txt'
path_to_results = 'Results'
#uncomment the model, you want to use
#my_model_name = 'VGG16'
#my_model_name = 'NASNetLarge'
#my_model_name = 'MobileNetV2'
#my_model_name = 'Xception'
my_model_name = 'DenseNet201'
output_features_dir = 'Features/'
output_features_file = output_features_dir + my_model_name + 'Features.txt'

#2350 * 0,1 = 235
sample_size = 235
batch_size = 64
epochs=4000
initial_learning_rate = 0.1
learning_rate_step = 2000
path_to_checkpoints = "checkpointsTest/"
#my_seed = 4321
hidden_layer_neurons_count = 128
output_layer_neurons_count = 4


print("******************************************")
print("Initial processing")
print("******************************************")
process_dataset(path_to_data, path_to_output, path_to_descriptions)

print("******************************************")
print("Starting Tensorflow")
print("******************************************")
physical_devices = enable_tensorflow()

print("******************************************")
print("Reading model")
print("******************************************")

my_model = None

if my_model_name == 'VGG16':
    my_model = create_model_VGG16()
if my_model_name == 'NASNetLarge':
    my_model = create_model_NASNetLarge()
if my_model_name == 'MobileNetV2':
    my_model = create_model_MobileNetV2()
if my_model_name == 'Xception':
    my_model = create_model_Xception()
if my_model_name == 'DenseNet201':
    my_model =  create_model_DenseNet201()


print("******************************************")
print("Features generation")
print("******************************************")
generate_features(my_model, my_model_name, path_to_descriptions, path_to_output, output_features_file, output_features_dir)

my_seeds = [0, 101, 542, 1011, 3333, 4321, 6000, 7777, 10111, 15151]
for my_seed in my_seeds:

    print("******************************************")
    print("Starting seed: " + str(my_seed))
    print("******************************************")

    print("******************************************")
    print("Transfer learning")
    print("******************************************")

    my_model_name_help = my_model_name + str(my_seed)
    #print(my_model_name_help)
    transfer_learning(sample_size, my_seed, batch_size, epochs, initial_learning_rate, learning_rate_step,
                           path_to_checkpoints, my_model_name_help, output_features_file, path_to_descriptions,
                           hidden_layer_neurons_count, output_layer_neurons_count)

    print("******************************************")
    print("Making predictions")
    print("******************************************")
    make_predictions(sample_size, hidden_layer_neurons_count, output_layer_neurons_count, my_model_name_help,
                         output_features_file, path_to_descriptions, path_to_checkpoints, path_to_results, my_seed)

    print("******************************************")
    print("Calculating confusion matrix")
    print("******************************************")
    calculate_confusion_matrix(sample_size, my_model_name_help, path_to_results, path_to_descriptions, my_seed)
