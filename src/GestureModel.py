#!/usr/bin/env python

# Luis Enrique Coronado Zuniga

# You are free to use, change, or redistribute the code in any way you wish
# but please maintain the name of the original author.
# This code comes with no warranty of any kind.

from Creator import*
from Classifier import*
from scipy import linalg
import matplotlib.pyplot as plt

class GestureModel():
     """This class is used to save the parameters of a model

          :param name_model: name of the model
          :param path: In which folder the model will be saved, or is saved?
          :param num_samples: number of samples by model
          :param th: threashold of the model
          :param create: If True then the model is created (GMR + GMM), if False the the model is only defined

          The .txt files used to create the models must be named as: *mod(i).txt*
          
          Where :math:`i=\{1,..,num\_samples\}`.
          """
     def __init__(self,name_model,path,num_samples,th,create = True):
          # Init the the type of the parameters
          print "New model = ", name_model
          
          self.name_model = name_model
          self.component = {}
          self.threashold = th
          self.Weights = {}
          self.path = path
          self.num_samples = num_samples
          self.files = []
          #Bool variable
          self.create = create 
          self.pathmodels =  self.path + self.name_model + '/models/'
          
          for k in range(1,self.num_samples+1):
               self.files.append(path + name_model + '/data/mod('+ str(k) + ').txt')

          if(self.create):
               self.createModel()


     def createModel(self):
          """Create a new model from a list of .txt files, using GMR +  GMM
          """
          print "Creating model"
          # New creator of models
          name_model = self.name_model
          g = Creator()
          #Read the  files	
          g.ReadFiles(self.files,[])
          g.CreateDatasets_Acc()

          # 1) Use Knn to obtain the number of cluster
          g.ObtainNumberOfCluster(save = True)
          
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


          #Save the model
          try:
               savetxt(self.pathmodels+ 'MuGravity.txt', gr_points,fmt='%.12f')
               savetxt(self.pathmodels+ 'SigmaGravity.txt', gr_sigma,fmt='%.12f')
               savetxt(self.pathmodels+ 'MuBody.txt', b_points,fmt='%.12f')
               savetxt(self.pathmodels+ 'SigmaBody.txt', b_sigma,fmt='%.12f')
          except:
               print "Error, folder not found"
                  
     def setModel(self,name,mean,sigma,threashold, weight = 0.5):
          """Set the parameters of the model

          :param name: name of the component (gravity or body)
          :param mean: mean of the model
          :param sigma: :math:`\\sigma` of the model
          :param threashold: threashold of the model
          :param weight: weights of the model
          """
          
          self.component[name] = [mean,sigma]
          self.threashold = threashold
          self.Weights[name] = weight
          print name
     

     def setWeight(self,name,value):
          """Set the weight of the model
               :param name: name of the component (gravity or body)
               :param value: new weight value 
          """
          self.Weights[name] = value

     def loadModel(self):
          """If a model was created before, then set the parameters of the model with this function"""

          #Load files
          self.gr_points = loadtxt(self.pathmodels+"MuGravity.txt")
          self.gr_sigma = loadtxt(self.pathmodels+"SigmaGravity.txt")

          self.b_points = loadtxt(self.pathmodels+"MuBody.txt")
          self.b_sigma = loadtxt(self.pathmodels+"SigmaBody.txt")

          self.setModel("gravity",self.gr_points, self.gr_sigma,self.threashold)
          self.setModel("body",self.b_points, self.b_sigma,self.threashold)


     def plotResults(self):
          """Plot the results of GMR + GMM used to create the model
          """
          import matplotlib.pyplot as plt
          gr_points =  self.gr_points
          gr_sig = self.gr_sigma
          b_points = self.b_points
          b_sig =  self.b_sigma
          name_model =  self.name_model

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
          


if __name__ == "__main__":
    import doctest
    doctest.testmod()


