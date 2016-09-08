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
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
from sklearn.mixture import GMM
from GestureModel import*
from Creator import*

class Classifier:

    def __init__(self):
        print "New classifier"


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
                window,numWritten = self.createWindow(one_sample, window, window_size, numWritten)
                if (numWritten >= window_size):
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

        g_Weight = w_g
        b_Weight = w_b

        print "Weight of the gravity", g_Weight
        print "Weight of the body", b_Weight

        return g_Weight, b_Weight

    def validate_from_file(self,sfile,delim = ' '):
        numWritten = 0
        data = genfromtxt(sfile, delimiter=delim)
        numSamples,m = data.shape
        print "Num samples", numSamples
        
        possibilities_vector = zeros((numSamples,self.nModels))
        for i in range (numSamples):
            one_sample = data[i]
            self.window,numWritten = self.createWindow(one_sample, self.window, self.window_size, numWritten)
            if (numWritten >= self.window_size):
                # Compute the acceleration components of the current window of samples
                gravity,body = self.analyzeActualWindow(self.window,self.window_size);
                possibilities = self.compareAll(self.models,self.nModels,gravity, body,self.window_size);
                for j in range(self.nModels):
                    possibilities_vector[i,j] = possibilities[0,j]
        return possibilities_vector
                    
    
    def createWindow(self,one_sample, window, window_size, numWritten):
        #Fill window if numWritten < window_size
        if numWritten < window_size:
            for j in range(3):
                window[numWritten,j] = one_sample[j];
            numWritten = numWritten + 1
        #shift window and update
        else:
            for i in range(window_size-1):
                for j in range(3):
                        window[i,j] = window[i+1,j];
            for j in range(3):
                window[window_size - 1,j] = one_sample[j];
            numWritten = numWritten + 1
        return window,numWritten

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



def plotResults(gr_points,gr_sig, b_points,b_sig,name_model):
    from scipy import linalg
    import matplotlib.pyplot as plt

    gr_points = gr_points.transpose()
    b_points = b_points.transpose()
     
    gr_sigma = []
    b_sigma = []

    n,m = gr_points.shape

    maximum = zeros((m))
    minimum = zeros((m))

    x = arange(0,m,1)

    for i in range(m):
        gr_sigma.append(gr_sig[i*3:i*3+3])
        b_sigma.append(b_sig[i*3:i*3+3])

    
    for i in range(m):
        sigma = 3.*linalg.sqrtm(gr_sigma[i])
        maximum[i] =  gr_points[0,i]+ sigma[0,0];
        minimum[i] =  gr_points[0,i]- sigma[0,0];

    fig2 = plt.figure()
    import matplotlib.pyplot as plt
    plt.fill_between(x, maximum, minimum,lw=2, alpha=0.5 )
    plt.plot(x, gr_points[0])
    plt.savefig(name_model+ "_gravity_x_axis.png")

    for i in range(m):
        sigma = 3.*linalg.sqrtm(gr_sigma[i])
        maximum[i] =  gr_points[1,i]+ sigma[1,1];
        minimum[i] =  gr_points[1,i]- sigma[1,1];

    fig3 = plt.figure()
    plt.fill_between(x, maximum, minimum,lw=2, alpha=0.5 )
    plt.plot(x, gr_points[1])
    plt.savefig(name_model+ "_gravity_y_axis.png")

    for i in range(m):
        sigma = 3.*linalg.sqrtm(gr_sigma[i])
        maximum[i] =  gr_points[2,i]+ sigma[2,2];
        minimum[i] =  gr_points[2,i]- sigma[2,2];

    fig3 = plt.figure()
    import matplotlib.pyplot as plt
    plt.fill_between(x, maximum, minimum,lw=2, alpha=0.5 )
    plt.plot(x, gr_points[2])
    plt.savefig(name_model+ "_gravity_z_axis.png")

    for i in range(m):
        sigma = 3.*linalg.sqrtm(b_sigma[i])
        maximum[i] =  b_points[0,i]+ sigma[0,0];
        minimum[i] =  b_points[0,i]- sigma[0,0];

    fig4 = plt.figure()
    import matplotlib.pyplot as plt
    plt.fill_between(x, maximum, minimum,lw=2, alpha=0.5 )
    plt.plot(x, b_points[0])
    plt.savefig(name_model+ "_body_x_axis.png")

    for i in range(m):
        sigma = 3.*linalg.sqrtm(b_sigma[i])
        maximum[i] =  b_points[1,i]+ sigma[1,1];
        minimum[i] =  b_points[1,i]- sigma[1,1];

    fig5 = plt.figure()
    import matplotlib.pyplot as plt
    plt.fill_between(x, maximum, minimum,lw=2, alpha=0.5 )
    plt.plot(x, b_points[1])
    plt.savefig(name_model+ "_body_axis.png")

    for i in range(m):
        sigma = 3.*linalg.sqrtm(b_sigma[i])
        maximum[i] =  b_points[2,i]+ sigma[2,2];
        minimum[i] =  b_points[2,i]- sigma[2,2];

    fig6 = plt.figure()
    import matplotlib.pyplot as plt
    plt.fill_between(x, maximum, minimum,lw=2, alpha=0.5 )
    plt.plot(x, b_points[2])
    plt.savefig(name_model+ "_body_z_axis.png")



def newModel(name,list_files,th=1):
    g = Creator()
    #Read the data
    g.ReadFiles(list_files,[])
    g.CreateDatasets_Acc()
    g.ObtainNumberOfCluster()
    gravity = g.gravity
    K_gravity = g.K_gravity
    body = g.body
    K_body = g.K_body

    # 2) define the number of points to be used in GMR
    #    (current settings allow for CONSTANT SPACING only)
    numPoints = amax(gravity[0,:]);
    scaling_factor = 10/10;
    numGMRPoints = math.ceil(numPoints*scaling_factor);

    # 3) perform Gaussian Mixture Modelling and Regression to retrieve the
    #   expected curve and associated covariance matrices for each feature

    gr_points, gr_sigma = g.GetExpected(gravity,K_gravity,numGMRPoints)
    b_points, b_sigma = g.GetExpected(body,K_body,numGMRPoints)
    

    savetxt(name+"MuGravity.txt", gr_points,fmt='%.12f')
    savetxt(name+"SigmaGravity.txt", gr_sigma,fmt='%.12f')
    savetxt(name+"MuBody.txt", b_points,fmt='%.12f')
    savetxt(name+"SigmaBody.txt", b_sigma,fmt='%.12f')


def loadModel(file_name, th=1,plot=True):

    #Load files
    gr_points = loadtxt(file_name+"MuGravity.txt")
    gr_sigma = loadtxt(file_name+"SigmaGravity.txt")

    b_points = loadtxt(file_name+"MuBody.txt")
    b_sigma = loadtxt(file_name+"SigmaBody.txt")

    #Add model
    gm = GestureModel()
    gm.addModel("gravity",gr_points, gr_sigma,th)
    gm.addModel("body",b_points, b_sigma,th)

    if plot == True:
        plotResults(gr_points,gr_sigma, b_points,b_sigma,file_name)
    
    return gm

#For ROS we must stay in the src folder

files1 = ["A/mod(1).txt","A/mod(2).txt","A/mod(3).txt","A/mod(4).txt","A/mod(5).txt","A/mod(6).txt" ]
#newModel("A",files1,th=3)
gm1 = loadModel("A",25)

files2 = ["P/mod(1).txt","P/mod(2).txt","P/mod(3).txt","P/mod(4).txt","P/mod(5).txt", "P/mod(6).txt"]
#newModel("P",files2,th=3)
gm2 = loadModel("P",20)

print "1"
v1 = Classifier()
v2 = Classifier()

w_g, w_b = v1.calculateW(files1,gm1)

print w_g, w_b

gm1.addWeight("gravity",w_g)
gm1.addWeight("body",w_b)


w_g, w_b = v1.calculateW(files2,gm2)

print w_g, w_b

gm2.addWeight("gravity",w_g)
gm2.addWeight("body",w_b)


 
v1.classify([gm1])
v2.classify([gm2])



sfile = "AP/acc(1).txt"
poss1 = v1.validate_from_file(sfile, ',')
poss2 = v2.validate_from_file(sfile, ',')

fig = plt.figure()

m,n = poss1.shape
print m,n

x = arange(0,m,1)
import matplotlib.pyplot as plt
plt.plot(x, poss1)

m,n = poss2.shape
print m,n

x = arange(0,m,1)
import matplotlib.pyplot as plt
plt.plot(x, poss2)


plt.savefig("result.png")


##print "\n\n 2"
##
##sfile = "D/mod(1).txt"
##v1.validate_from_file(sfile)






