#! /usr/bin/env python
import rospy, math
import sys, termios, tty, select, os
from geometry_msgs.msg import Twist
from std_msgs.msg import UInt16
import sys
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
import signal
from GestureModel import*
from Creator import*
from Classifier import*
from numpy import*
from numpy.linalg import*
from scipy import interpolate
from scipy.signal import filtfilt, lfilter
from scipy.signal import medfilt
from scipy.signal import filter_design as ifd
from scipy.stats import multivariate_normal
import scipy.spatial
import os
import time
import std_msgs.msg

th1 = .5
th2 = .5

def callback(data):
	"""Callback function"""
	x= data.position.x
	y= data.position.y
	z= data.position.z
	#print "Data received ----------- \n", x, y, z
	#start = time.time()
	#for j in range(len(name_models)):
	#poss[0] =  l_class[0].online_validation(x,y,z)
	#poss[1] =  l_class[1].online_validation(x,y,z)
	poss[2] =  l_class[2].online_validation(x,y,z)
	#poss[3] =  l_class[3].online_validation(x,y,z)

	#print >>sys.stderr, poss
  
	if(poss[0]>th1):
		print "avanti"
	if(poss[1]>th1):
		print "Stop"
	if(poss[2]>th1):
		print "Boton"
	if(poss[3]>th1):
		print "Aplauso"
	#update(1,0)
 #   if(p2>th2):
#	print "stop"
#	update(0,0)
    #done = time.time()
    #elapsed = done - start
	


def signal_handler(signal, frame):
	"""Signal handler of the data """
	print('Signal Handler, you pressed Ctrl+C!')
	print('Server will be closed')
 	sys.exit(0)


def loadModel(file_name, th=1, plot=True):

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

name_models = ['A','P','B','S']
num_samples = [10,8,14,9]
th = [25,20,35,40]
create_models = False
list_files = []

#Create a list of the list of files for each model

print "Defining files"
i = 0
for name in name_models:
    files = []
    for k in range(1,num_samples[i]+1):
        files.append('Models/' + name + '/data/mod('+ str(k) + ').txt')
    list_files.append(files)
    i = i + 1
    
    
#Create the models and save the list of files for calculate the weigths
if(create_models == True):
    print "Creating models"
    i = 0
    for model in name_models:
        print list_files[i]
        newModel(model,list_files[i])
        i = i + 1
        
list_models = []

print "Loading models"
#Load the models
for j in range(len(name_models)):
     #For the moment don't put True is there are more that 2 models in Ubuntu
     gm = loadModel(name_models[j],th[j],False)
     list_models.append(gm)


print "Calculating weigths"

#Used to calculate the weights
v0 = Classifier()

for j in range(len(name_models)):
    print "\nFor model " + name_models[j] + ":"
    w_g, w_b = v0.calculateW(list_files[j],list_models[j])
    list_models[j].addWeight("gravity",w_g)
    list_models[j].addWeight("body",w_b)
    

print "\n Init classifers"

#List of classifiers
l_class = []

for j in range(len(name_models)):
     l_class.append(Classifier())

print "Give the model to each classifier"


poss = []
for j in range(len(name_models)):
    l_class[j].classify(list_models[j])
    poss.append(0)

print "Validation"
  

signal.signal(signal.SIGINT, signal_handler)
update_rate = 20
rospy.init_node('keyboard_teleop')
rospy.Subscriber('/wearami_acc', Pose, callback)
#pub_twist = rospy.Publisher('/cmd_vel', Twist,queue_size = 5)
pub_twist = rospy.Publisher('turtle1/cmd_vel', Twist, queue_size=5)
try:
	r = rospy.Rate(update_rate) # Hz
	while not rospy.is_shutdown():
	        #update()
	        r.sleep()
except rospy.exceptions.ROSInterruptException:
	pass



