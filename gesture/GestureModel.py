#!/usr/bin/env python

# Luis Enrique Coronado Zuniga

# You are free to use, change, or redistribute the code in any way you wish
# but please maintain the name of the original author.
# This code comes with no warranty of any kind.

from gesture import*
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
          :param create: If True then the model is created, if False the the model is only defined
          :param dtype: Type of data. "3IMU_acc" -> IMU 3D data
          :param feature_extraction: Bool variable

          The .txt files used to create the models must be named as: *mod(i).txt*
          
          Where :math:`i=\{1,..,num\_samples\}`.
          """
     def __init__(self,name_model,path_to_load,path_to_save,num_samples):
          # Init the the type of the parameters

          print
          print "*************  Model = ", name_model, "***********"
          print

          # Name of the social cue model
          self.name_model = name_model
          
          # This social signal can have diferent components, a dictionary is used to save the
          # information of each component
          self.component = {}
          self.Weights = {}
          self.num_samples = num_samples
          print num_samples
          self.files = []

          # Diferent path where is loaded and saved
          self.path_data =  path_to_load  + self.name_model
          self.path_models =  path_to_save  + self.name_model
          print path_to_load + name_model

          # Open the files
          for k in range(1,num_samples+1):
               self.files.append(self.path_data + '/mod('+ str(k) + ').txt')

          self.gesture_model = newModel()

          #Read the  files	
          self.gesture_model.ReadFiles(self.files)

          if not os.path.exists(self.path_models):
               os.makedirs(self.path_models)


               
     def buildModel(self, mtype = "GMM", dtype = "3IMU_acc",feature_extraction = True, th = 1, value = "100"):
          """Build a new Gesture model from a list of .txt files
               :param mtype: type of model "GMM" or "DNN"
               :param dtype: Type of data. "3IMU_acc" -> IMU 3D data
               :param feature_extraction: Bool variable
               :param th: threashold (only GMM)
               :param value: number of samples for each training set (for DNN)
          """

                    
          self.threashold = th
          
          print "Building a Gesture model"
               

          # Create a GMM model
          if mtype == "GMM":              
               if dtype == "3IMU_acc":
                    print "Type Model = GMM , Type data= 3IMU_acc"
                    if feature_extraction:
                         self.gesture_model.extract3D_acceleration_features()
                         self.createGMM_model_f()
                    else:
                         self.gesture_model.IMU2matrix()
                         self.createGMM_model()

          # Create a DNN model
          if mtype == "DNN":              
               if dtype == "3IMU_acc":
                    print "Type Model = DNN , Type data= 3IMU_acc"
                    if feature_extraction:
                         self.gesture_model.extract3D_acceleration_features()
                         self.createDNN_model_f(value)
                    else:
                         self.gesture_model.IMU2matrix()
                         self.createDNN_model(value)

     # ------------------------------------------------- DNN ---------------------------------------------------------------


     def list2KerasIMU(self, value):
          """Convert a list of datafiles from IMU 3d informtion to a matrix for be used in Keras
          """
          n_data = self.gesture_model.n_data
          
          x = ones((1,n_data))*self.gesture_model.datafiles[0][0].transpose()
          y = ones((1,n_data))*self.gesture_model.datafiles[0][1].transpose()
          z = ones((1,n_data))*self.gesture_model.datafiles[0][2].transpose()

          if(value <= n_data):
               x = x[:,0:value-1]
               y = y[:,0:value-1]
               z = z[:,0:value-1]

          else:              

               additional = value - n_data -1
               x = concatenate((x,zeros((1,additional))), axis=1)
               y = concatenate((y,zeros((1,additional))), axis=1)
               z = concatenate((z,zeros((1,additional))), axis=1)

          data = concatenate((x,y), axis=1)
          data = concatenate((data, z), axis=1)
          cdata = data

          i = 1


          while (i < self.gesture_model.nfiles):
               x = ones((1,n_data))*self.gesture_model.datafiles[i][0].transpose()
               y = ones((1,n_data))*self.gesture_model.datafiles[i][1].transpose()
               z = ones((1,n_data))*self.gesture_model.datafiles[i][2].transpose()
               if(value <= n_data):
                    x = x[:,0:value-1]
                    y = y[:,0:value-1]
                    z = z[:,0:value-1]
               else:
                    x = concatenate((x,zeros((1,additional))), axis=1)
                    y = concatenate((y,zeros((1,additional))), axis=1)
                    z = concatenate((z,zeros((1,additional))), axis=1)

               data = concatenate((x,y), axis=1)
               data = concatenate((data, z), axis=1)
               cdata = concatenate((cdata, data), axis=0)
               i = i + 1
   
          return cdata

     def createDNN_model(self, value):
          self.dnn_dataset = self.list2KerasIMU(value)





     def concatenate_in_vector(data):
          print data
          
               

     # ------------------------------------------------- GMM ---------------------------------------------------------------
     def createGMM_model(self,):

          gmm_model = self.gesture_model

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
               
     def createGMM_model_f(self):

          gmm_model = self.gesture_model


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

     def loadModel(self, dtype = "3IMU_acc", threashold = 100, feature_extraction = True):
          """If a model was created before, then set the parameters of the model with this function
               :param dtype: Type of data. "3IMU_acc" -> IMU 3D data
               :param feature_extraction: Bool variable
               :param threashold: threashold value
          """

          #Load files
          self.threashold = threashold
          
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


