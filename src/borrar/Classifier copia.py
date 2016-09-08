#!/usr/bin/env python

"""@See preprocessed data
"""
from numpy import arange, sin, pi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from matplotlib.path import Path
import matplotlib.patches as patches
from numpy import*
from numpy.linalg import*
from scipy import interpolate
from scipy.signal import filtfilt, lfilter
from scipy.signal import medfilt
from scipy.signal import filter_design as ifd
from scipy.stats import multivariate_normal
import scipy.spatial
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
from sklearn.mixture import GMM
from GestureModel import*

class Classifier:

    def __init__(self):
        print "New classifier"
        self.numWritten = 0


    def classify(self,models):
        self.models = models
        #This in the class
        self.nModels = len(models)
        models_size = zeros((1,self.nModels))
        for i in range(self.nModels):
            models_size[0,i],m = (models[i].component["gravity"][0].shape)
        print "models size = ", models_size
        """This returns a list"""
        self.window_size = int(nanmin(models_size))
        print "window size", self.window_size
        self.window = zeros((self.window_size,3))


    def calculateW(self,sfiles,model):

        window_size,m = (model.component["gravity"][0].shape)
        print "models size = ", window_size
        window = zeros((window_size,3))

        totalG = 0
        totalB = 0

        
        k = 0
        for file in sfiles:
            numWritten = 0
            data = genfromtxt(file, delimiter=' ')
            numSamples,m = data.shape

            distanceG = 0
            distanceB = 0
            
            for j in range (numSamples):
                one_sample = data[j]
                window = self.createWindow(one_sample, window, window_size)
                self.numWritten = self.numWritten + 1
                if (self.numWritten >= window_size):
                    # Compute the acceleration components of the current window of samples
                    gravity,body = self.analyzeActualWindow(window,window_size)
                    distanceG = distanceG + self.compare(model,"gravity",gravity,window_size, "mh")
                    distanceB = distanceB + self.compare(model,"body",body,window_size,"mh")
                    k = k + 1

        #New implemntation
            totalG = totalG + distanceG
            totalB = totalB + distanceB

        mean_g =  totalG/k
        mean_b = totalB/k

        w_g = mean_g/(mean_g + mean_b)
        w_b = mean_b/(mean_g + mean_b)

        g_Weight = 1- w_g
        b_Weight = 1-  w_b

        print "Weight of the gravity", g_Weight
        print "Weight of the body", b_Weight
        
        #Reset the value
        self.numWritten = 0

        return g_Weight, b_Weight

    def validate_from_file(self,sfile,delim = ' '):
        numWritten = 0
        data = genfromtxt(sfile, delimiter= delim)
        numSamples,m = data.shape
        print "Num samples", numSamples
        
        possibilities_vector = zeros((numSamples,self.nModels))
        for i in range (numSamples):
            one_sample = data[i]
            self.window = self.createWindow(one_sample, self.window, self.window_size)
            if (numWritten >= self.window_size):
                # Compute the acceleration components of the current window of samples
                gravity,body = self.analyzeActualWindow(self.window,self.window_size);
                possibilities = self.compareAll(self.models,self.nModels,gravity, body,self.window_size);
                print possibilities
                for j in range(self.nModels):
                    possibilities_vector[i,j] = possibilities[0,j]
                    
    def online_validation(self,x,y,z):
        one_sample = array([x,y,z])
        self.window = self.createWindow(one_sample, self.window, self.window_size)
        self.numWritten = self.numWritten + 1
        if (self.numWritten >= self.window_size):
            # Compute the acceleration components of the current window of samples
            gravity,body = self.analyzeActualWindow(self.window,self.window_size);
            possibilities = self.compareAll(self.models,self.nModels,gravity, body,self.window_size);
            #print possibilities
        return possibilities
                    
    
    def createWindow(self,one_sample, window, window_size):
        #Fill window if numWritten < window_size
        if self.numWritten < window_size:
            for j in range(3):
                window[self.numWritten,j] = one_sample[j];
        #shift window and update
        else:
            for i in range(window_size-1):
                for j in range(3):
                        window[i,j] = window[i+1,j];
            for j in range(3):
                window[window_size - 1,j] = one_sample[j];
        return window

    def analyzeActualWindow(self,window,numSamples):
        """ function [gravity body] = AnalyzeActualWindow(window,numSamples)
        %
        % AnalyzeActualWindow separates the gravity and body acceleration features
        % contained in the window of real-time acceleration data, by first reducing
        % the noise on the raw data with a median filter and then discriminating
        % between the features with a low-pass IIR filter."""

        #REDUCE THE NOISE ON THE SIGNALS BY MEDIAN FILTERING
        n = 3  #order of the median filter
        x_axis = medfilt(window[:,0],n)
        y_axis = medfilt(window[:,1],n)
        z_axis = medfilt(window[:,2],n)

        #APPLY IIR FILTER TO GET THE GRAVITY COMPONENTS
        #IIR filter parameters (all frequencies are in Hz)
        Fs = 32;            # sampling frequency
        Fpass = 0.25;       # passband frequency
        Fstop = 2;          # stopband frequency
        Apass = 0.001;      # passband ripple (dB)
        Astop = 100;        # stopband attenuation (dB)
        match = 'pass';     # band to match exactly
        #Create the IIR filter


        # iirdesign agruements
        Wip = (Fpass)/(Fs/2)
        Wis = (Fstop+1e6)/(Fs/2)
        Rp = Apass             # passband ripple
        As = Astop            # stopband attenuation

        # The iirdesign takes passband, stopband, passband ripple, 
        # and stop attenuation. 
        bb, ab = ifd.iirdesign(Wip, Wis, Rp, As, ftype='cheby1')

        #Gravity components
        g1 = lfilter(bb,ab,x_axis)
        g2 = lfilter(bb,ab,y_axis)
        g3 = lfilter(bb,ab,z_axis)


        #COMPUTE THE BODY-ACCELERATION COMPONENTS BY SUBTRACTION  (PREGUNTA)
        gravity = zeros((numSamples,3));
        body = zeros((numSamples,3));

        i=0
        while(i < numSamples):
            gravity[i,0] = g1[i];
            gravity[i,1] = g2[i];
            gravity[i,2] = g3[i];
            body[i,0] = x_axis[i] - g1[i];
            body[i,1] = y_axis[i] - g2[i];
            body[i,2] = z_axis[i] - g3[i];
            i = i + 1

        #COMPUTE THE BODY-ACCELERATION COMPONENTS BY SUBTRACTION
        return gravity, body

    def compare(self,model,string_component,component,window_size, distance_type = "mh"):
        """Returns the distance from the model"""
        mean =  model.component[string_component][0] #0 is the mean
        sigma =  model.component[string_component][1] #1 is sigma

        dist = zeros((1,window_size));
        if(distance_type == "mh"): #Mahalanobis distance
            for i in range(window_size):
                dist[0,i] = scipy.spatial.distance.mahalanobis(mean[i],component[i],sigma[i*3:((i+1)*3)])
        distance = sum(dist)/window_size #mean
        return distance
        

    def compareOne(self,gravity,body,model,window_size):
        
        distanceG = self.compare(model,"gravity",gravity,window_size, "mh")
        distanceB = self.compare(model,"body",body,window_size,"mh")

        #print "\n G= ", distanceG, " B=", distanceB
        #print model.Weights["gravity"], model.Weights["body"]
        overall = model.Weights["gravity"]*distanceG + model.Weights["body"]*distanceB
        #print "overall", overall
        return overall


    def compareAll (self,models,nModels,gravity,body,window_size):
        distance =  zeros((1,nModels))
        possibilities = zeros((1,nModels))
        for i in range(nModels):
            distance[0,i] =  self.compareOne(gravity, body, models[i],window_size)

        """ compute the possibility value of each model
        (mapping of the likelihoods from [0..threshold(i)] to [1..0]"""

        for i in range(nModels):
            possibilities[0,i] = 1 - distance[0,i]/models[i].threashold;
            if (possibilities[0,i] < 0):
                possibilities[0,i] = 0
        return possibilities

