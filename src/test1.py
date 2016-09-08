#!/usr/bin/env python

"""@See preprocessed data
"""
from numpy import*
import matplotlib.pyplot as plt
from GestureModel import*
from Creator import*
from Classifier import*


path_models = 'Models/'

string1 = 'arriba'
create_models = False
list_files = []
string2 = 'alado'
string3 = 'yo'

gmB1 = GestureModel(string1,path_models,6,400,create_models)
gmB2 = GestureModel(string2,path_models,5,400,create_models)
gmB3 = GestureModel(string3,path_models,5,400,create_models)

#Create a list of models
list_models = [gmB1,gmB2,gmB3]


print "Loading models"
for model in list_models:
    model.loadModel()
    #model.plotResults()

"""print "body points"
print gmB1.b_points

print "gravity points"
print gmB1.gr_points

print "body sigma"
print gmB1.b_sigma

print "gravity sigma"
print gmB1.gr_points"""

from Recognition import*
r =  Recognition(list_models)

#Calculate best weights
print "Calculating weigths"
r.calculate_Weights()

print "Validation"

sfile = "validation/t1.txt"
r.recognition_from_files(sfile)


sfile = "validation/t2.txt"
r.recognition_from_files(sfile)


sfile = "validation/t3.txt"
r.recognition_from_files(sfile)

sfile = "validation/mix.txt"
r.recognition_from_files(sfile)
"""
print "Finish ..."

"""





