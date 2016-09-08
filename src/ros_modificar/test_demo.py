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

    start = time.time()
    p1 = v1.online_validation(x,y,z)
    p2 = v2.online_validation(x,y,z)
    if(p1>th1):
	print "avanti"
	update(1,0)
    if(p2>th2):
	print "stop"
	update(0,0)
    done = time.time()
    elapsed = done - start

#Put in a class
def loadModel(file_name,th=1,plot=False):

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



def signal_handler(signal, frame):
	"""Signal handler of the data """
	print('Signal Handler, you pressed Ctrl+C!')
	print('Server will be closed')
 	sys.exit(0)


def update(x,z):
	if rospy.is_shutdown():
		return
	twist = Twist()
	twist.linear.x = x
	twist.angular.z = z
	pub_twist.publish(twist)


#Model HMPdetector
files1 = ["A/mod(1).txt","A/mod(2).txt","A/mod(3).txt","A/mod(4).txt","A/mod(5).txt","A/mod(6).txt" ]
#newModel("A",files1,th=3)
gm1 = loadModel("A",25)

files2 = ["P/mod(1).txt","P/mod(2).txt","P/mod(3).txt","P/mod(4).txt","P/mod(5).txt", "P/mod(6).txt"]
#newModel("P",files2,th=3)
gm2 = loadModel("P",22)
print "Models added"

#gm1 and gm2 are the models


#Tune the modedels W
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

print "Models tunned"

#Set models to classify
v1.classify(gm1)
v2.classify(gm2)

print "Models ready..."
  

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



