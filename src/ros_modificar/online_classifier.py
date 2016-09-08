#!/usr/bin/env python

import sys
import rospy
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


def signal_handler(signal, frame):
    """Signal handler of the data """
    print('Signal Handler, you pressed Ctrl+C!')
    print('Server will be closed')
    sys.exit(0)


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

cont1 = 0
cont2 = 0

def callback(data):
    """Callback function"""
    x= data.position.x
    y= data.position.y
    z= data.position.z
    #print "Data received ----------- \n", x, y, z

    start = time.time()
    p1 = v1.online_validation(x,y,z)
    p2 = v2.online_validation(x,y,z)
    #print p1, "  ", p2
    if(p1>th1):
	print "avanti"
	pub.publish(1)
        #cont1 = cont1 + 1
    if(p2>th2):
	print "stop"
	pub.publish(0)
        #cont2 = cont2 + 1
    done = time.time()
    elapsed = done - start

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


files1 = ["A/mod(1).txt","A/mod(2).txt","A/mod(3).txt","A/mod(4).txt","A/mod(5).txt","A/mod(6).txt" ]
#newModel("A",files1,th=3)
gm1 = loadModel("A",25)

files2 = ["P/mod(1).txt","P/mod(2).txt","P/mod(3).txt","P/mod(4).txt","P/mod(5).txt", "P/mod(6).txt"]
#newModel("P",files2,th=3)
gm2 = loadModel("P",22)

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


v1.classify(gm1)
v2.classify(gm2)


cont1 = 0
cont2 = 0
th1 = .5
th2 = .5

print "ROS classifier ready"

#ROS ----------

rospy.init_node('online_classifier', anonymous=True)
rate = rospy.Rate(100) # period
signal.signal(signal.SIGINT, signal_handler)
rospy.Subscriber('/wearami_acc', Pose, callback)
pub = rospy.Publisher('/hmp_gesture', std_msgs.msg.Int32,queue_size =2 )
rospy.spin()








