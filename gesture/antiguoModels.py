#!/usr/bin/env python

# Luis Enrique Coronado Zuniga

# You are free to use, change, or redistribute the code in any way you wish
# but please maintain the name of the original author.
# This code comes with no warranty of any kind.

from nep_gauss import*
from scipy import linalg
import matplotlib.pyplot as plt
import os

class GestureModel():
     """This class is used to save the parameters of a model

          :param name_model: name of the model
          :param path_to_load: In which folder the data is loaded?
          :param path_to_save: In which folder the model will be saved?
          :param num_samples: number of samples by model
          :param th: threashold of the model
          :param create: If True then the model is created (GMR + GMM), if False the the model is only defined
          :param dtype: Type of data. "3IMU_acc" -> IMU 3D data
          :param  feature_extraction: Bool variable

          The .txt files used to create the models must be named as: *mod(i).txt*
          
          Where :math:`i=\{1,..,num\_samples\}`.
          """
     def __init__(self,name_model,path_to_load,path_to_save,num_samples,th,create = True, dtype = "3IMU_acc", feature_extraction = True):
          # Init the the type of the parameters

          print
          print "*************  Model = ", name_model, "***********"
          print

          # Name of the social cue model
          self.name_model = name_model
          
          # This social signal can have diferent components, a dictionary is used to save the
          # information of each component
          self.component = {}
          
          self.threashold = th
          self.Weights = {}
          self.num_samples = num_samples
          self.files = []
          self.create = create

          # Diferent path where is loaded and saved
          self.path_data =  path_to_load  + self.name_model
          self.path_models =  path_to_save  + self.name_model
          print path_to_load + name_model

          # Open the files
          for k in range(1,num_samples):
               self.files.append(self.path_data + '/mod('+ str(k) + ').txt')

          # Create a GMM model if self.create == True
          if(self.create):
               #self.createModel()
               self.buildModel(dtype, feature_extraction)

               
     def buildModel(self, dtype = "3IMU_acc", feature_extraction = True):
          """Build a new GMM model from a list of .txt files
               :param dtype: Type of data. "3IMU_acc" -> IMU 3D data
               :param  feature_extraction: Bool variable
          """
          print "Building the GMM model"

          # New creator of models
          gmm_model = newModel()

          #Read the  files	
          gmm_model.ReadFiles(self.files)

          if not os.path.exists(self.path_models):
               os.makedirs(self.path_models)
               
          if dtype == "3IMU_acc":
               print "Type = 3IMU_acc model"
               if feature_extraction:
                    gmm_model.extract3D_acceleration_features()
                    self.create3IMU_acc_model_f(gmm_model)
               else:
                    gmm_model.IMU2matrix()
                    self.create3IMU_acc_model(gmm_model)
               

     def create3IMU_acc_model(self,gmm_model):


          # 1) Use Knn to obtain the number of cluster
          # TO IMPROVE
          gmm_model.ObtainNumberOfCluster(dtype = "3IMU_acc", feature_extraction = False, save = True, path = self.path_models)
          
          acc = gmm_model.acc
          K_acc = gmm_model.K_acc

          # 2) define the number of points to be used in GMR
          #    (current settings allow for CONSTANT SPACING only)
          numPoints = amax(acc[0,:]);
          scaling_factor = 10/10;
          numGMRPoints = math.ceil(numPoints*scaling_factor);

          # 3) perform Gaussian Mixture Modelling and Regression to retrieve the
          #   expected curve and associated covariance matrices for each feature

          acc_points, acc_sigma = gmm_model.GetExpected(acc,K_acc,numGMRPoints)


          #Save the model
          try:
               savetxt(self.path_models+ '/MuIMUacc.txt', acc_points,fmt='%.12f')
               savetxt(self.path_models+ '/SigmaIMUacc.txt', acc_sigma,fmt='%.12f')
          except:
               print "Error, folder not found"
               
     def create3IMU_acc_model_f(self,gmm_model):


          # 1) Use Knn to obtain the number of cluster
          # TO IMPROVE
          gmm_model.ObtainNumberOfCluster(dtype = "3IMU_acc", feature_extraction = True, save = True, path = self.path_models)
          
          gravity = gmm_model.gravity
          K_gravity = gmm_model.K_gravity
          body = gmm_model.body
          K_body = gmm_model.K_body

          # 2) define the number of points to be used in GMR
          #    (current settings allow for CONSTANT SPACING only)
          numPoints = amax(gravity[0,:]);
          scaling_factor = 10/10;
          numGMRPoints = math.ceil(numPoints*scaling_factor);

          # 3) perform Gaussian Mixture Modelling and Regression to retrieve the
          #   expected curve and associated covariance matrices for each feature

          gr_points, gr_sigma = gmm_model.GetExpected(gravity,K_gravity,numGMRPoints)
          b_points, b_sigma = gmm_model.GetExpected(body,K_body,numGMRPoints)


          #Save the model
          try:
               savetxt(self.path_models+ '/MuGravity.txt', gr_points,fmt='%.12f')
               savetxt(self.path_models+ '/SigmaGravity.txt', gr_sigma,fmt='%.12f')
               savetxt(self.path_models+ '/MuBody.txt', b_points,fmt='%.12f')
               savetxt(self.path_models+ '/SigmaBody.txt', b_sigma,fmt='%.12f')
          except:
               print "Error, folder not found"

     # DELETE this function
##     def createModel(self):
##          """Create a new model from a list of .txt files, using GMM + GMR
##          """
##          print "Building the GMM model"
##
##          # New creator of models
##          g = Creator()
##
##          #Read the  files	
##          g.ReadFiles(self.files,[])
##          g.CreateDatasets_Acc()
##          
##          
##          if not os.path.exists(self.path_models):
##              os.makedirs(self.path_models)
##
##          # 1) Use Knn to obtain the number of cluster
##          # TO IMPROVE
##          g.ObtainNumberOfCluster(save = True, path = self.path_models)
##          
##          gravity = g.gravity
##          K_gravity = g.K_gravity
##          body = g.body
##          K_body = g.K_body
##
##          # 2) define the number of points to be used in GMR
##          #    (current settings allow for CONSTANT SPACING only)
##          numPoints = amax(gravity[0,:]);
##          scaling_factor = 10/10;
##          numGMRPoints = math.ceil(numPoints*scaling_factor);
##
##          # 3) perform Gaussian Mixture Modelling and Regression to retrieve the
##          #   expected curve and associated covariance matrices for each feature
##
##          gr_points, gr_sigma = g.GetExpected(gravity,K_gravity,numGMRPoints)
##          b_points, b_sigma = g.GetExpected(body,K_body,numGMRPoints)
##
##
##          #Save the model
##          try:
##               savetxt(self.path_models+ '/MuGravity.txt', gr_points,fmt='%.12f')
##               savetxt(self.path_models+ '/SigmaGravity.txt', gr_sigma,fmt='%.12f')
##               savetxt(self.path_models+ '/MuBody.txt', b_points,fmt='%.12f')
##               savetxt(self.path_models+ '/SigmaBody.txt', b_sigma,fmt='%.12f')
##          except:
##               print "Error, folder not found"
                  
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

     def loadModel(self, dtype = "3IMU_acc", feature_extraction = True):
          """If a model was created before, then set the parameters of the model with this function
               :param dtype: Type of data. "3IMU_acc" -> IMU 3D data
               :param  feature_extraction: Bool variable
          """

          #Load files
          
          if dtype == "3IMU_acc":
               print "Type = 3IMU_acc model"
               if feature_extraction:
                    print "Using features of Gravity and Body"
                    self.gr_points = loadtxt(self.path_models+"/MuGravity.txt")
                    self.gr_sigma = loadtxt(self.path_models+"/SigmaGravity.txt")

                    self.b_points = loadtxt(self.path_models+"/MuBody.txt")
                    self.b_sigma = loadtxt(self.path_models+"/SigmaBody.txt")

                    self.setModel("gravity",self.gr_points, self.gr_sigma,self.threashold)
                    self.setModel("body",self.b_points, self.b_sigma,self.threashold)

               else:
                    print "No features of Gravity and Body"
                    self.acc_points = loadtxt(self.path_models+"/MuIMUacc.txt")
                    self.acc_sigma = loadtxt(self.path_models+"/SigmaIMUacc.txt")
                    self.setModel("acc",self.acc_points, self.acc_sigma,self.threashold)

                    


     def plotResults_IMU_acc_f(self):
          """Plot the results of GMR + GMM used to create the model (Gravity and Body)
          """
          import matplotlib.pyplot as plt
          gr_points =  self.gr_points
          gr_sig = self.gr_sigma
          b_points = self.b_points
          b_sig =  self.b_sigma

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
          plt.savefig(self.path_models+ "/_gravity_x_axis.png")
          plt.close('all')

          for i in range(m):
             sigma = 3.*linalg.sqrtm(gr_sigma[i])
             maximum[i] =  gr_points[1,i]+ sigma[1,1];
             minimum[i] =  gr_points[1,i]- sigma[1,1];

          fig3 = plt.figure()
          plt.fill_between(x, maximum, minimum,lw=2, alpha=0.5 )
          plt.plot(x, gr_points[1])
          plt.savefig(self.path_models+ "/_gravity_y_axis.png")
          plt.close('all')

          for i in range(m):
             sigma = 3.*linalg.sqrtm(gr_sigma[i])
             maximum[i] =  gr_points[2,i]+ sigma[2,2];
             minimum[i] =  gr_points[2,i]- sigma[2,2];

          fig3 = plt.figure()
          import matplotlib.pyplot as plt
          plt.fill_between(x, maximum, minimum,lw=2, alpha=0.5 )
          plt.plot(x, gr_points[2])
          plt.savefig(self.path_models+ "/_gravity_z_axis.png")
          plt.close('all')

          for i in range(m):
             sigma = 3.*linalg.sqrtm(b_sigma[i])
             maximum[i] =  b_points[0,i]+ sigma[0,0];
             minimum[i] =  b_points[0,i]- sigma[0,0];

          fig4 = plt.figure()
          import matplotlib.pyplot as plt
          plt.fill_between(x, maximum, minimum,lw=2, alpha=0.5 )
          plt.plot(x, b_points[0])
          plt.savefig(self.path_models+ "/_body_x_axis.png")
          plt.close('all')

          for i in range(m):
             sigma = 3.*linalg.sqrtm(b_sigma[i])
             maximum[i] =  b_points[1,i]+ sigma[1,1];
             minimum[i] =  b_points[1,i]- sigma[1,1];

          fig5 = plt.figure()
          import matplotlib.pyplot as plt
          plt.fill_between(x, maximum, minimum,lw=2, alpha=0.5 )
          plt.plot(x, b_points[1])
          plt.savefig(self.path_models+ "/_body_axis.png")
          plt.close('all')

          for i in range(m):
             sigma = 3.*linalg.sqrtm(b_sigma[i])
             maximum[i] =  b_points[2,i]+ sigma[2,2];
             minimum[i] =  b_points[2,i]- sigma[2,2];

          fig6 = plt.figure()
          import matplotlib.pyplot as plt
          plt.fill_between(x, maximum, minimum,lw=2, alpha=0.5 )
          plt.plot(x, b_points[2])
          plt.savefig(self.path_models+ "/_body_z_axis.png")
          plt.close('all')

     def plotResults_IMU_acc(self):
          """Plot the results of GMR + GMM used to create the model (Only acceleration data)
          """
          import matplotlib.pyplot as plt
          gr_points =  self.acc_points
          gr_sig = self.acc_sigma


          gr_points = gr_points.transpose()
     

          gr_sigma = []
         

          n,m = gr_points.shape

          maximum = zeros((m))
          minimum = zeros((m))

          x = arange(0,m,1)

          for i in range(m):
             gr_sigma.append(gr_sig[i*3:i*3+3])
     


          for i in range(m):
             sigma = 3.*linalg.sqrtm(gr_sigma[i])
             maximum[i] =  gr_points[0,i]+ sigma[0,0];
             minimum[i] =  gr_points[0,i]- sigma[0,0];

          fig2 = plt.figure()
          import matplotlib.pyplot as plt
          plt.fill_between(x, maximum, minimum,lw=2, alpha=0.5 )
          plt.plot(x, gr_points[0])
          plt.savefig(self.path_models+ "/_acc_x_axis.png")
          plt.close('all')

          for i in range(m):
             sigma = 3.*linalg.sqrtm(gr_sigma[i])
             maximum[i] =  gr_points[1,i]+ sigma[1,1];
             minimum[i] =  gr_points[1,i]- sigma[1,1];

          fig3 = plt.figure()
          plt.fill_between(x, maximum, minimum,lw=2, alpha=0.5 )
          plt.plot(x, gr_points[1])
          plt.savefig(self.path_models+ "/_acc_y_axis.png")
          plt.close('all')

          for i in range(m):
             sigma = 3.*linalg.sqrtm(gr_sigma[i])
             maximum[i] =  gr_points[2,i]+ sigma[2,2];
             minimum[i] =  gr_points[2,i]- sigma[2,2];

          fig3 = plt.figure()
          import matplotlib.pyplot as plt
          plt.fill_between(x, maximum, minimum,lw=2, alpha=0.5 )
          plt.plot(x, gr_points[2])
          plt.savefig(self.path_models+ "/_acc_z_axis.png")
          plt.close('all')

          


if __name__ == "__main__":
    import doctest
    doctest.testmod()


