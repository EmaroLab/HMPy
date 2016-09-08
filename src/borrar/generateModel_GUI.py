"""@See preprocessed data
"""
import Tkinter as tk
import ttk
from Tkinter import StringVar
import tkFileDialog
from ttk import Frame, Style
from tkMessageBox import*
from numpy import arange, sin, pi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from numpy import*
from numpy.linalg import*
from scipy import interpolate
from scipy.signal import filtfilt, lfilter
from scipy.signal import medfilt
from scipy.signal import filter_design as ifd
from scipy.stats import multivariate_normal
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


class generateModel(Frame):
    """Interfaz class"""
    def __init__(self, master):
        self.master = master
        """tkinter variable"""
        self.initUI()
        self.lista = []
        self.noisy_x = []
        self.noisy_y = []
        self.noisy_z = []
        self.x_set = []
        self.y_set = []
        self.z_set = []
        """list of acceleration files"""


    def initUI(self):
        """Definition of the GUI"""
        master = self.master
        self.master.title("See data")
        self.openButton = tk.Button(master, text = 'Select Data', width = 30 , height = 3, command = self.ReadFiles).grid(row=1,column=0, padx = 10, pady = 10)
        self.seeallButton = tk.Button(master, text = 'Plot data', width = 30 , height = 3, command = self.see).grid(row=2,column=0, padx = 10, pady = 10)
        
 
##        self.labelname1 = tk.Label(master, text='Axis')
##        self.labelname1.grid(row=3,column=0)
##        self.numberS = StringVar()
##        self.numbertxt = tk.Entry(master, width = 20, textvariable = self.numberS )
##        self.numbertxt.insert(0, "1")
##        self.numbertxt.grid(row=4,column=0)

        self.figure = Figure(figsize=(4, 4), dpi=100)
        """Figure variable to plot with matplotlib"""
        self.canvas = FigureCanvasTkAgg(self.figure, master)
        # pack and grid dont get along in Mac OS
        if (os == 'Windows'):
            self.toolbar = NavigationToolbar2TkAgg(self.canvas, master)
        self.canvas.get_tk_widget().grid(row=1,column=1,rowspan = 5,padx = 10, pady = 10)
        if (os == 'Windows'):
            self.toolbar.grid(row=6,column=1)
        self.canvas.draw()

        self.CreateDatasetsButton = tk.Button(master, text = 'Create Datasets', width = 30 , height = 3, command = self.CreateDatasets).grid(row=4,column=0, padx = 10, pady = 10)
        self.CreateDatasetsButton = tk.Button(master, text = 'Tune K and Compute Expected Curves', width = 30 , height = 3, command = self.GetExpectedCurves).grid(row=5,column=0, padx = 10, pady = 10)


    def see(self):
        """Plot all the data together"""
        self.plotData();

    #1
    def ReadFiles(self):
        """Read the data from txt files"""
        try:
            import matplotlib.pyplot as plt
            self.names = tkFileDialog.askopenfilenames(title='Choose acceleration files')
            self.num = 0
            for name in self.names:
                self.data = genfromtxt(name, delimiter=' ')
                self.lista.append(self.data)
                noisy_x = self.lista[self.num].transpose()[0]
                noisy_y = self.lista[self.num].transpose()[1]
                noisy_z = self.lista[self.num].transpose()[2]
                self.noisy_x.append(self.lista[self.num].transpose()[0])
                self.noisy_y.append(self.lista[self.num].transpose()[0])
                self.noisy_z.append(self.lista[self.num].transpose()[0])
                self.num = self.num + 1
                n = 3  #order of the smedian filter
                x_set = medfilt(noisy_x,n)
                y_set = medfilt(noisy_y,n)
                z_set = medfilt(noisy_z,n)
                self.x_set.append(x_set)
                self.y_set.append(y_set)
                self.z_set.append(z_set)
                self.numSamples,m = self.lista[0].shape
                
        except:
            showerror("Error in the files ")

    def plotData(self):
        #Plot aw data or data with median filter
        number = 1
        a1 = self.figure.add_subplot(311)
        a2 = self.figure.add_subplot(312)
        a3 = self.figure.add_subplot(313)
        a1.clear()
        a2.clear()
        a3.clear()
        """0 for plot raw data"""
        if(number == 0):
            for i in range(self.num):
                n,m = self.lista[i].shape
                x = arange(0,n,1)
                a1.plot(x, self.lista[i].transpose()[0])
                a2.plot(x, self.lista[i].transpose()[1])
                a3.plot(x, self.lista[i].transpose()[2])
        """1 for plot data with median filter"""
        if(number == 1):
            for i in range(self.num):
                n = size(self.x_set[0])
                x = arange(0,n,1)
                a1.plot(x, self.x_set[i])
                a2.plot(x, self.y_set[i])
                a3.plot(x, self.z_set[i])
        a1.grid(True)
        a2.grid(True)
        a3.grid(True)
        self.canvas.draw()

    #2
    def CreateDatasets(self):
        """ CreateDatasets computes the gravity and body acceleration components
            of the trials given in the [*_set]s by calling the function GetComponents
            for each trial and reshapes the results into one set of gravity
            components and one set of body acceleration components according to the
            requirements of Gaussian Mixture Modelling.

        Output:
           gravity --> dataset of the gravity components along the axes with the
                       time indexes and the acceleration values on 4 rows
                       (row2 -> x_axis, row3 -> y_axis, row4 -> z_axis)
                       and all of the trials concatenated one after the other
           body --> dataset of the body acc. components along the axes with the
                       time indexes and the acceleration values on 4 rows
                       (row2 -> x_axis, row3 -> y_axis, row4 -> z_axis)
                       and all of the trials concatenated one after the other"""

        #% SEPARATE THE GRAVITY AND BODY-MOTION ACCELERATION COMPONENTS
        #Obtain the number of files 
        numFiles = len(self.x_set)
        gravity_trial,body_trial = self.GetComponents(self.x_set[0], self.y_set[0], self.z_set[0])
        self.shortNumSamples, m = gravity_trial.shape
        #print self.shortNumSamples
        #initial values of the dataset arrays
        n = self.shortNumSamples
        
        time = ones((1,n))*arange(0,self.shortNumSamples,1)
        g_x_s = ones((1,n))*gravity_trial[0:self.shortNumSamples,0].transpose()
        g_y_s = ones((1,n))*gravity_trial[0:self.shortNumSamples,1].transpose()
        g_z_s = ones((1,n))*gravity_trial[0:self.shortNumSamples,2].transpose()
        b_x_s = ones((1,n))*body_trial[0:self.shortNumSamples,0].transpose()
        b_y_s = ones((1,n))*body_trial[0:self.shortNumSamples,1].transpose()
        b_z_s = ones((1,n))*body_trial[0:self.shortNumSamples,2].transpose()

        i = 1
        while(i < self.num):
            gravity_trial,body_trial = self.GetComponents(self.x_set[i], self.y_set[i], self.z_set[i])
            # CREATE THE DATASETS FOR THE GMMs
            timec = ones((1,n))*arange(0,self.shortNumSamples,1)
            time = concatenate((time,timec),axis=1)
            g_x_s = concatenate((g_x_s,ones((1,n))*gravity_trial[0:self.shortNumSamples,0].transpose()),axis=1)
            g_y_s = concatenate((g_y_s,ones((1,n))*gravity_trial[0:self.shortNumSamples,1].transpose()),axis=1)
            g_z_s = concatenate((g_z_s,ones((1,n))*gravity_trial[0:self.shortNumSamples,2].transpose()),axis=1)
            b_x_s= concatenate((b_x_s,ones((1,n))*body_trial[0:self.shortNumSamples,0].transpose()),axis=1)
            b_y_s = concatenate((b_y_s,ones((1,n))*body_trial[0:self.shortNumSamples,1].transpose()),axis=1)
            b_z_s = concatenate((b_z_s,ones((1,n))*body_trial[0:self.shortNumSamples,2].transpose()),axis=1)
            i = i +1

        

        gravity = concatenate((time,g_x_s), axis = 0);
        gravity = concatenate((gravity, g_y_s), axis = 0);
        self.gravity = concatenate((gravity, g_z_s), axis = 0);

        body = concatenate((time,b_x_s), axis = 0);
        body = concatenate((body,b_y_s), axis = 0);
        self.body = concatenate((body,b_z_s), axis = 0);

    #2.1
    def GetComponents(self, x_axis, y_axis, z_axis):
        """ GetComponents discriminates between gravity and body acceleration by
            applying an infinite impulse response (IIR) filter to the raw
            acceleration data (one trial) given in input."""
        #APPLY IIR FILTER TO GET THE GRAVITY COMPONENTS
        #IIR filter parameters (all frequencies are in Hz)
        Fs = 32;            # sampling frequency
        Fpass = 0.25;       # passband frequency
        Fstop = 2;          # stopband frequency
        Apass = 0.001;      # passband ripple (dB)
        Astop = 100;        # stopband attenuation (dB)
        match = 'pass';     # band to match exactly
        delay = 64;         # delay (# samples) introduced by filtering
        #Create the IIR filter

        

        # iirdesign agruements
        Wip = (Fpass)/(Fs/2)
        Wis = (Fstop+1e6)/(Fs/2)
        Rp = Apass             # passband ripple
        As = Astop            # stopband attenuation

        # The iirdesign takes passband, stopband, passband ripple, 
        # and stop attenuation. 
        bb, ab = ifd.iirdesign(Wip, Wis, Rp, As, ftype='cheby1')
        
        g1 = lfilter(bb,ab,x_axis)
        g2 = lfilter(bb,ab,y_axis)
        g3 = lfilter(bb,ab,z_axis)

        #COMPUTE THE BODY-ACCELERATION COMPONENTS BY SUBTRACTION  (PREGUNTA)
        gravity = zeros((self.numSamples -delay,3));
        body = zeros((self.numSamples -delay,3));

        i = 0
        while(i < self.numSamples-delay):
            #shift & reshape gravity to reduce the delaying effect of filtering
            gravity[i,0] = g1[i+delay];
            gravity[i,1] = g2[i+delay];
            gravity[i,2] = g3[i+delay];
            
            body[i,0] = x_axis[i] - gravity[i,0];
            body[i,1] = y_axis[i] - gravity[i,1];
            body[i,2] = z_axis[i] - gravity[i,2];
            i = i + 1

        #COMPUTE THE BODY-ACCELERATION COMPONENTS BY SUBTRACTION

 
        return gravity, body


    #3
    def GetExpectedCurves(self):
        """Compuet the expected curve for each dataset"""
        path = tkFileDialog.asksaveasfilename()
        #1) determine the number of Gaussians to be used in the GMM
        
        self.K_gravity = self.TuneK(self.gravity,100,'gravity',path)
        print "K gravity = ", self.K_gravity, "\n"
        self.K_body = self.TuneK(self.body,100,'body',path)
        print "K body = ", self.K_body, "\n"

        #Gaussian Mixter

        # 2) define the number of points to be used in GMR
        #    (current settings allow for CONSTANT SPACING only)
        numPoints = amax(self.gravity[0,:]);
        scaling_factor = 10/10;
        numGMRPoints = math.ceil(numPoints*scaling_factor);
        
        # 3) perform Gaussian Mixture Modelling and Regression to retrieve the
        #   expected curve and associated covariance matrices for each feature
        
        self.GetExpected(self.gravity,self.K_gravity,numGMRPoints)
        self.GetExpected(self.body,self.K_body,numGMRPoints)

    def TuneK(self,set_, maxK, name, path):
        """ TuneK determines the optimal number of clusters to be used to cluster
            the given [set] with K-means algorithm. It cycles from K = 2 to [maxK].
            The optimization criterion adopted is a variant of the elbow method: at
            each iteration TuneK computes the silhouette values of the clusters
            determined by the K-means algorithm and compares them with the values
            obtained at the previous iteration. When the quality of the
            clustering falls below a fixed threshold, TuneK stops.

            Input:
            set --> either the gravity or the body acc. dataset retrived from
                CreateDatasets
            maxK --> maximum number of clusters to be used to cluster the given
                dataset. Default: 1/2 of the number of data-points
                composing the dataset

            Output:
            Koptimal --> optimal number of clusters to be used to cluster the data
                of the given dataset """

        # DETERMINE THE OPTIMAL NUMBER OF CLUSTERS (K) FOR THE GIVEN DATASET
        # tuning parameters
        threshold = 0.69  # threshold on the FITNESS of the current clustering
        minK = 2         # initial number of clusters to be used
        #first step is outside of the loop to have meaningful initial values
        data = set_.transpose()#[:,1:]
        n_samples, n_features = data.shape
        print   "samples= ", n_samples, "features", n_features
        #n_digits = len(unique(digits.target))
        #labels = digits.target

        print(79 * '_')
        print(name + '\n')
        return self.bench_k_means(data,name,path)

    def bench_k_means(self, data, name, path):

    #In this example the silhouette analysis is used to choose an optimal value for n_clusters.
    #Bad pick for the given data due to the presence of clusters with
    #below average silhouette scores and also due to wide fluctuations in the size of the silhouette plots.

        threshold = 0.69;

        #t0 = time()
        X = data
        cmin = 2
        cmax = 50

        
        for n_clusters in range(cmin,cmax):
            # Create a subplot with 1 row and 2 columns

            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.set_size_inches(18, 7)
            
            
            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 but in this example all
            # lie within [-0.1, 1]
            ax1.set_xlim([-0.1, 1])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(X)
            #cluster_labels = clusterer.fit(X)

            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(X, cluster_labels, metric='sqeuclidean')
            print("For n_clusters =", n_clusters,
                  "The average silhouette_score is :", silhouette_avg)

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(X, cluster_labels)
            #print sample_silhouette_values

            Koptimal = n_clusters
            #if (Koptimal == maxK):
                #print('MATLAB:noConvergence','Failed to converge to the optimal K: increase maxK.')

            if(silhouette_avg < threshold):
                return (Koptimal)


            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.spectral(float(i) / n_clusters)
                ax1.fill_betweenx(arange(y_lower, y_upper),
                                  0, ith_cluster_silhouette_values,
                                  facecolor=color, edgecolor=color, alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhoutte score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # 2nd Plot showing the actual clusters formed
            colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
            ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                        c=colors)

            # Labeling the clusters
            centers = clusterer.cluster_centers_
            # Draw white circles at cluster centers
            ax2.scatter(centers[:, 0], centers[:, 1],
                        marker='o', c="white", alpha=1, s=200)

            for i, c in enumerate(centers):
                ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

            ax2.set_title("The visualization of the clustered data.")
            ax2.set_xlabel("Feature space for the 1st feature")
            ax2.set_ylabel("Feature space for the 2nd feature")

            plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                          "with n_clusters = %d" % n_clusters),
                         fontsize=14, fontweight='bold')
            #Debug Quit comment
            #plt.savefig(path + str(name) + "_c_"+ str(n_clusters) + '.png')

        return Koptimal


    def GetExpected(self,set_,K,numGMRPoints):
        """GetExpected performs Gaussian Mixture Modeling (GMM) and Gaussian Mixture
        Regression (GMR) over the given dataset. It returns the expected curve
        (expected mean for each point, computed over the values given in the
        [set]) and associated set of covariance matrices defining the "model"
        of the given dataset. 
    
        Input:
        set --> either the gravity or the body acc. dataset retrived from
               CreateDatasets
        K --> optimal number of clusters to be used to cluster the data
               of the given dataset retrieved from TuneK
        numGMRPoints --> number of data points composing the expected curves to
               be computed by GMR """
        
        # PARAMETERS OF THE GMM+GMR PROCEDURE
        numVar = 4;            # number of variables in the system (time & 3 accelerations)
        m,numData = set_.shape;# number of points in the dataset

        #Inicializar valores, existe la opcion.
        #priors, mu, sigma = self.InitializeGMM(set_,K,numVar)

        
        gmm = GMM(n_components=K,covariance_type='full')
        gmm.fit(set_.transpose())

        priors = gmm.weights_
        mu = gmm.means_
        sigma = gmm.covars_
                
        #print "piors:", priors
        #print "mu:", mu
        #print "sigma:", sigma
 
        #self.TrainGMM(K,set_,priors,mu,sigma,numVar,numData)

        #APPLY GAUSSIAN MIXTURE REGRESSION TO FIND THE EXPECTED CURVE
        #define the points to be used for the regression
        #(assumption: CONSTANT SPACING)
        expData1 = ceil(np.linspace(amin(set_[0,:],),amax(set_[0,:]), num=numGMRPoints));
        #Input dimension = 0, output = 1,2,...
        self.RetrieveModel(K,priors,mu,sigma,expData1,0,np.arange(1,numVar),numVar)
 
        

    def InitializeGMM(self,set_,K,numVar):
        """InitializeGMM initializes the parameters to be used in the Gaussian
        Mixture Model (a-priori probabilities, mean and standard deviation for
        each Gaussian) by using the K-means clustering algorithm on the given
        dataset.

        Input:
          set --> dataset of the points to be modelled (clustered with K-means)
          K --> number of Gaussian functions to be used in the GM model =
                number of clusters to be used in the K-means clustering
          numVar --> number of independent variables reported in the dataset
                     (time; acc. along x; acc. along y; acc. along z)

        Output:
          priors --> a-priori probabilities of all of the clusters
          mu --> centroids of the clusters = means of the Gaussians
          sigma --> standard deviation of the points from the clusters' centroids
                    (Euclidean distance) = covariance matrices of the Gaussians"""
        print set_
        X = set_.transpose()
        clusterer = KMeans(n_clusters=K, random_state=10)
        cluster_labels = clusterer.fit_predict(X)
        centroids = clusterer.cluster_centers_
        mu = centroids.transpose()
        priors = zeros((1,K))
        sigma = zeros((numVar,numVar,K))
        for i in range(K):
            temp_id = where(cluster_labels==i)
            priors[0,i] = size(temp_id[0]);
            #add a tiny variance to avoid numerical instability
            temp = concatenate((set_[:,temp_id[0]], set_[:,temp_id[0]]),axis=1)
            covariance =  cov(temp) + 1E-5*identity(numVar)
            sigma[:,:,i] =  covariance
        priors = priors / sum(priors);
        return priors, mu, sigma
    

    def TrainGMM(self,K,set_,priors,mu,sigma,numVar,numData):
        """
        TrainGMM computes the Gaussian Mixture Model of the given dataset
        starting from the initial values provided by the function InitializeGMM
        and training the modelling parameters with the Expectation-Maximization
        algorithm. The function returns the a-priori probabilities, means and
        covariance matrices of the Gaussians.

        Input:
        K --> number of Gaussian functions to be used in the GM model"""

                
        #PARAMETERS OF THE E-M ALGORITHM
        log_likelihood_threshold = 1e-10;   # threshold on the log likelihood
        log_likelihood_old = -1000000000;      # initial log likelihood

        # APPLY THE E-M ALGORITHM TO TUNE THE GAUSSIANS' PARAMETERS

        #while (rue):
            #EXPECTATION step
            #for i in range(K):
                # Compute the probability of each point to belong to the actual GM
                # model (probability density function of the point) --> p(point)
        multivariate_normal.pdf(set_.transpose(),mu[0,:],sigma[:,:,0])
           

    def RetrieveModel(self,K,priors,mu,sigma,points,in_,out,numVar):
        numData = size(points)
        pdf_point = zeros((numData,K))
        exp_point_k = zeros((numVar-1,numData,K))
        exp_sigma_k = {}
        for i in range(K):
            # compute the probability of each point to belong to the actual GM
            # model (probability density function of the point) --> p(point)
            pdf_point_temp = multivariate_normal.pdf(points,mu[i,in_],sigma[i,in_,in_])
            #compute p(Gaussians) * p(point|Gaussians)
            pdf_point[:,i] = priors[i]* pdf_point_temp
        #estimate the parameters beta
        beta = pdf_point/tile(sum(pdf_point),[1,K])

        for j in range (K):
            temp = (ones((numData,1))*mu[j,out]).transpose()+(ones((1,numVar-1))*(sigma[j,out,in_]*1/(sigma[j,in_,in_]))).transpose()*(points-tile(mu[j,in_],[1,numData]))
            exp_point_k[:,:,j] = temp

        beta_tmp = reshape(beta,(1,numData,K))
        #print tile(beta_tmp,[size(out),1,1]).shape
        exp_point_k2 = tile(beta_tmp,[size(out),1,1])*exp_point_k;
        #compute the set of expected means
        expMeans = sum(exp_point_k2,2)

        for j in range (K):
            #exp_point_k[j,:,:]=
            temp = (ones((1,numVar-1))*(sigma[j,out,out])) - (ones((1,numVar-1))*(sigma[j,out,in_]*1/(sigma[j,in_,in_]))).transpose()*sigma[j,in_,out]
            exp_sigma_k[j] = temp
            
        expSigma = {}
        for i in range (numData):
            expSigma[i] =  zeros((numVar-1,numVar-1))
            for j in range (K):
                        expSigma[i] = expSigma[i] + beta[i,j]* beta[i,j]*exp_sigma_k[j]
        return expMeans, expSigma

            
        
        

                              
    
        
    

