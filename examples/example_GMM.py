#!/usr/bin/env python

# Luis Enrique Coronado Zuniga
# You are free to use, change, or redistribute the code in any way you wish
# but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Descrption: ---------------------------------
# In this example GMM models of 3 gestures are created
# Then GMR is used for recognition of this gestures
# ---------------------------------------------
#%%
from numpy import*
from gesture import GestureModel, Recognition
import os

dir_path = os.getcwd()
print "Current path in ", dir_path

# ----------- Configuration of the models----------

# Do not extract grevity and body features
features = False

#Define the name of the gesture
name_model1 = "hand_up"
name_model2 = "hand_pendulum"
name_model3 = "press_button"

# Define the path to save the GMM models
path_to_save = dir_path + "/data_examples/Gaussian_Models/"

# Define the path of the gesture folders with the training examples
path_to_load = dir_path + "/data_examples/acceleration_aligned/"

# Define the name of the files in the datatest, example <mod(1)>.txt, example <mod(2)>.txt , ....
files_id = "mod"

# Define a threashlod
th = 400

# Crate a new model
create_models = True

# Define the number of training examples
training_examples1 = 10
training_examples2 = 10
training_examples3 = 10

# Create new gesture models clases.
# Note: if the error `ValueError: need more than 2 values to unpack` appear 
# maybe the number of samples of the training examples a specifc gestues are not the same for all
gest1 = GestureModel(name_model1,path_to_load,path_to_save,training_examples1)
gest2 = GestureModel(name_model2,path_to_load,path_to_save,training_examples2)
gest3 = GestureModel(name_model3,path_to_load,path_to_save,training_examples3)

# Train GMM the models
if(create_models == True):
    gest1.buildModel("GMM", "3IMU_acc", features, th)
    gest2.buildModel("GMM", "3IMU_acc", features, th)
    gest3.buildModel("GMM", "3IMU_acc", features, th)

#Create a list of models
list_models = [gest1,gest2,gest3]

# Load the GMM models
print "Loading models ..."
th = [80,125,115]
i = 0
for model in list_models:
    model.loadModel("3IMU_acc", th[i], features)
    i = i +  1
    if(features):
        model.plotResults_IMU_acc_f()
    else:
        model.plotResults_IMU_acc()

# %%

# New recognition class for Recognition
r =  Recognition(list_models,"3IMU_acc", features )

# Define the raw datasets paths (without aligned)
path_to_validate = dir_path + "/data_examples/acceleration_raw/"

name_models = [name_model1,name_model2,name_model3]
print "Validation"

# Test examples no used for training
n_initial = 20
n_final = 20


# Example in ipython
%matplotlib inline 

for n_model in name_models:
    for n_data in range(n_initial,n_final+1):
        sfile = n_model + "/acc" + str(n_data) +   ".txt"
        print sfile
        r.recognition_from_files(path_to_validate, sfile, False, n_data)



