#!/usr/bin/env python

# Luis Enrique Coronado Zuniga

# You are free to use, change, or redistribute the code in any way you wish
# but please maintain the name of the original author.
# This code comes with no warranty of any kind.

"""@See preprocessed data
"""
from numpy import*
import matplotlib.pyplot as plt
from gesture import*

class Recognition():
    """Class used to recognize a gesture
        :param list_models: list of GestureModel
        :param save: bool parameter that indicates if the plots are saved
        :param path: path where the plots will be saved

        """
    def __init__(self,list_models,dtype = "3IMU_acc", feature_extraction = True):
        self.list_classifiers = []
        self.list_models = list_models
        self.dtype = dtype
        self.feature_extraction = feature_extraction

        #Create classifiers
        for j in range(len(list_models)):
            self.list_classifiers.append(Classifier())

        #Give the model to each classifier

        for j in range(len(list_models)):
            self.list_classifiers[j].classify(list_models[j],self.dtype,self.feature_extraction)


    def calculate_Weights(self):
        """Calculate weitghs of a list of model, using the classifier class"""

        v0 = Classifier()
        for model in self.list_models:
            print "\nWeigths for model " + model.name_model + ":"
            w_g, w_b = v0.calculateW(model.files,model)
            model.setWeight("gravity",w_g)
            model.setWeight("body",w_b)

    def recognition_from_files(self,path, sfile,save = False, n = 1):
        """Class used to recognize a gesture
        :param sfile: txt file, x y and z data separated by ','
        :param save: bool
        :param number: number of the sample

        """
        list_models = self.list_models

        fig = plt.figure(n)
        for j in range(len(list_models)):
            poss =  self.list_classifiers[j].validate_from_file(path + sfile, ',')
            m,n = poss.shape
            x = arange(0,m,1)
            plt.plot(x, poss,'-',label= list_models[j].name_model)
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                   ncol=2, mode="expand", borderaxespad=0.)
        plt.show()
        if(save):
            plt.savefig("result" + str(n) + ".png")

    def online_classification(self,model_number,x,y,z):
        """Class used to recognize a gesture
        :param model_number: number of the model
        :param x: x data
        :param y: y pdata
        :param z: z data

        """
        list_models = self.list_models
        return self.list_classifiers[model_number].online_validation(x,y,z)



if __name__ == "__main__":
    import doctest
    doctest.testmod()
