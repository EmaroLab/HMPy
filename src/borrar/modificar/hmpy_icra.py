#!/usr/bin/env python

"""@See preprocessed data
"""
from numpy import*
import matplotlib.pyplot as plt
from GestureModel import*
from Creator import*
from Classifier import*



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


#NOTE: Add path
def newModel(name,files):
    g = Creator()
    #Read the data	
        
    g.ReadFiles(files,[])
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



name_models = ['B','S1']
num_samples = [14,9]
th = [20,10]
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

l_class = []

for j in range(len(name_models)):
     l_class.append(Classifier())

print "Give the model to each classifier"

for j in range(len(name_models)):
    l_class[j].classify(list_models[j])

print "Validation"

sfile = "validation/acc(3).txt"

import matplotlib.pyplot as plt

fig = plt.figure()
for j in range(len(name_models)):
    poss =  l_class[j].validate_from_file(sfile, ',')
    m,n = poss.shape
    x = arange(0,m,1)
    plt.plot(x, poss,'o',label= name_models[j])
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
    
plt.savefig("result.png")

print "Finish ..."




