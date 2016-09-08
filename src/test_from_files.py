#!/usr/bin/env python

"""@See preprocessed data
"""
from numpy import*
import matplotlib.pyplot as plt
from GestureModel import*
from Creator import*
from Classifier import*


path_models = 'Models/'

name_models = ['B1','S5']
create_models = True
list_files = []

gmB1 = GestureModel('B1',path_models,20,25,create_models)
gmS5 = GestureModel('S5',path_models,14,80,create_models)

#Create a list of models
list_models = [gmB1,gmS5]

if(create_models==False):
    print "Loading models"
    for model in list_models:
        model.loadModel()

from Recognition import*
r =  Recognition(list_models)

#Calculate best weights
print "Calculating weigths"
r.calculate_Weights()

print "Validation"

sfile = "validation/acc(1).txt"
r.recognition_from_files(sfile)

print "Finish ..."




