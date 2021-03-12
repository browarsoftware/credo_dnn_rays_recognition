# Convolutional neural network (CNN)-based approach to the problems of categorization and artefact reduction of cosmic ray images obtained from CMOS sensors used in mobile phones

Source code for DNN-based cosmic rays recognition. 


This is a convolutional neural network (CNN)-based approach to the problems of categorization and artefact reduction of cosmic ray images obtained from CMOS 
sensors used in mobile phones. As artefacts, we understand all images that cannot be attributed to particles’ passage through sensor but rather result from the 
deficiencies of the registration procedure. The proposed deep neural network is composed of a pretrained CNN and neural-network-based approximator, 
which models the uncertainty of image class assignment. The network was trained using a transfer learning approach with a mean squared error loss function. 
We evaluated our approach on a data set containing 2350 images labelled by five judges. The most accurate results were obtained using the VGG16 CNN architecture; 
the recognition rate (RR) was 85.79% ± 2.24% with a mean squared error (MSE) of 0.03 ± 0.00. 
After applying the proposed threshold scheme to eliminate less probable class assignments, we obtained a RR of 96.95% ± 1.38% for a threshold of 0.9,
which left about 62.60% ± 2.88% of the overall data. 

Please cite as:

Hachaj, T.; Bibrzycki, Ł.; Piekarczyk, M. Recognition of Cosmic Ray Images Obtained from CMOS Sensors Used in Mobile Phones by Approximation of Uncertain Class Assignment with Deep Convolutional Neural Network. Sensors 2021, 21, 1963. https://doi.org/10.3390/s21061963 

Full text avilable at: https://www.mdpi.com/1424-8220/21/6/1963

Data source is from the CREDO project: https://credo.science/?lang=pl_pl
If you have some questions please contact tomasz.hachaj at up.krakow.pl
