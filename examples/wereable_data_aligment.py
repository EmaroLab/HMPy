#!/usr/bin/env python

# Luis Enrique Coronado Zuniga
# You are free to use, change, or redistribute the code in any way you wish
# but please maintain the name of the original author.
# This code comes with no warranty of any kind.


#%%

# Descrption: -------------------------------------------------------
# In this example a set of gestures dataset from 3-axis accelerometer 
# are aligned for posterior processing.
# --------------------------------------------------------------------


# Import Aligment class
import os
from gesture import Aligment
from matplotlib import pyplot as plt

# Example in ipython
%matplotlib inline 

# Get current path
c_path = os.getcwd()


# Define the path of the main folder that contain the gestures.
# In this example this folder is called `wereable_acc_models_raw`:
path = c_path + "/data_examples/acceleration_raw"
print "Main path of the gestures defined in:",  path

# We consider that the datasets of each gesture are separated 
# in folders with the name of the gesture

# We define the name of the gesture
name_model = "hand_up"

# We consider that each training example are in txt files with the format of
# `file_id + number_of_example + .txt`
# For this example the txt files of each gesture have a file_id of:
files_id = "acc"

# We need to define the max number of examples
training_examples = 15

#%%

# Define the dimention (x = 0,y = 1, or z =2) in which the aligment will be based. 
# If there are not good results try with other axis.
axis = 0

# Create a new class A for the aligment of the data
A = Aligment(path, name_model,files_id,training_examples,axis)

# Aligment using dtw
offsets = A.dtw_aligment()

#See the offsets betweem data
print offsets

newpath =  c_path + "/data_examples/acceleration_aligned"

# Save the aligned data, create a new folder in `newpath` with the name of the gesture

# Each training example must have the same number of samples for a specific gesture
# The we need to define the limits of the samples to be cutted
cut_left = 50
cut_right = 130
# TO DO:  How to do this automatically, without manual inspection?

# This function save a cut the data.
# Also shows if the data is saved correctly, otherwise also shows what to do
new_data = A.save(cut_left,cut_right,offsets,newpath)

# Show original data, figure 1
A.plotData()
plt.show()

# Show aligned data, figure 2
A.plotDataAligned()
plt.show()