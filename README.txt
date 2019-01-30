-----------------------------------------------------------------------------------
Code for "Learning to Fuse 2D and 3D Image Cues for Monocular Body Pose Estimation"
-----------------------------------------------------------------------------------

This folder includes code for the approach presented in

Bugra Tekin, Pablo Marquez-Neila, Mathieu Salzmann, Pascal Fua, "Learning to Fuse 2D and 3D Image Cues for Monocular Body Pose Estimation", ICCV 2017. (https://arxiv.org/pdf/1611.05708.pdf)

The code is implemented using the Theano library and tested on a Linux machine with a Python 2.7.

------------
Requirements
------------

- Theano
- cuDNN
- numpy
- scipy
- opencv-python
- h5py
- matplotlib

----
Data
----

We provide a subset of our preprocessed data for the Human3.6m data. Because of the sheer size of the dataset, we provide data only for an example action (Walking Dog) and also subsample the data by 5. We further provide a model trained on this subsampled data for testing. Please follow the instructions below to download and place the data and the models:

wget -O data/h36m_data_hg_preds_A_14_train.h5 https://drive.switch.ch/index.php/s/hGTYthjvOj5YG80/download
wget -O data/h36m_data_hg_preds_A_14_val.h5 https://drive.switch.ch/index.php/s/0jmlJCugid6kH1x/download
wget -O data/h36m_data_imgs_A_14_train.h5 https://drive.switch.ch/index.php/s/yHfjKcqDZohFIHz/download
wget -O data/h36m_data_imgs_A_14_val.h5 https://drive.switch.ch/index.php/s/RbLihBKpppHIH4L/download
wget -O model/h36m_model_A_14.npz https://drive.switch.ch/index.php/s/YswnuwALNoKhZGj/download

Instead you can also manually download the data and the models with the provided links and place them respectively in the "data" and "model" folders.

--------
Training
--------

You can train the network by running

python train.py

As the provided data is a subsampled version of the actual full data, the final training accuracy might be slightly different. You can experiment with different network structures by adding a custom architecture in "src/net/trainable_fusion.py" function. 

-------
Testing
-------

You can test and visualize the 3D human pose predictions by a model pretrained on the provided data using:

python test.py

-------
Contact
-------

For any question or bug report, please feel free to contact me at: bugra <dot> tekin <at> epfl <dot> ch

