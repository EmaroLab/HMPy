#!/usr/bin/env python

"""@See preprocessed data
"""

from numpy import arange, sin, pi
from numpy import*
from numpy.linalg import*
from dtw import dtw



class Aligment:
    """Class used to aligment of acceleration data()
        :param path: path of the files
        :param name: name of the files
        :param num_samples: number of files
    """

    def __init__(self, name_model, num_samples, axis = 0, filter = False):
        path = 'Models/'
        self.files = []
        self.axis = axis
        self.lista = []
        self.offsets = []
        self.name_model = name_model
        for k in range(1,num_samples+1):
            if(filter):
                self.files.append(path + name_model + '/filter/acc('+ str(k) + ').txt')
            else:
                self.files.append(path + name_model + '/raw/acc('+ str(k) + ').txt')

        self.n = 0
        for name in self.files:
            self.data = genfromtxt(name, delimiter=',')
            self.lista.append(self.data)
            self.offsets.append(0)
            self.n = self.n + 1



    def dtw_aligment(self):
        """Aligment of data using DTW algorithm
        :return offsets: list of offset betweem the firt sample an the others
        """
        x = self.lista[0][0:,self.axis].reshape(-1, 1)
        for k in range (1,self.n):
            y = self.lista[k][0:,self.axis].reshape(-1, 1)
            dist, cost, acc, path = dtw(x, y, dist=lambda x, y: norm(x - y, ord=1))
            map_x = path[0]
            map_y = path[1]
            counts = bincount(path[0])
            self.offsets[k] = int(mean(map_x - map_y))

        return self.offsets

    def save(self,limit_i, limit_r,offsets,lista):
        """Save the data """
        new_data = []
        path_models = 'Models/'
        filename = path_models + self.name_model + "/data/mod"
        print "Saving data ... \n"
        num = len(lista)
        for i in range(num):
            samples = int(limit_r) - int(limit_i)
            offset = int(offsets[i])
            init = limit_i - offset
            n,m = lista[i].shape
                
            if limit_i < offset:
                print "Error in saving data, sample", i, "increase left limit"
            elif n + offset <  limit_r:
                print "Error in saving data, sample", i, "decrease rigth limit"
            else:
                savetxt(filename + "(" + str(i+1) +').txt', lista[i][init:init+samples,:] , delimiter=' ', fmt='%f')
                new_data.append(lista[i][init:init+samples,:])
                print "Saving data", i, "OK"
                
        return new_data


    def plotData(self,lista):
        import matplotlib.pyplot as plt
        """Plot the data"""
        num = len(lista)
        fig = plt.figure()

        for i in range(num):
            n,m = lista[i].shape
            x = arange(0,n,1)
            axes = fig.add_subplot(311)
            axes.plot(x, lista[i].transpose()[0])
            axes.grid(True)
            axes = fig.add_subplot(312)
            axes.plot(x, lista[i].transpose()[1])
            axes.grid(True)
            axes = fig.add_subplot(313)
            axes.plot(x, lista[i].transpose()[2])
            axes.grid(True)

    def plotDataAligned(self):
        import matplotlib.pyplot as plt
        """Plot the data"""
        num = len(self.lista)
        fig = plt.figure()
        

        for i in range(num):

            x = self.lista[i][0:,0].reshape(-1, 1)
            y = self.lista[i][0:,1].reshape(-1, 1)
            z = self.lista[i][0:,2].reshape(-1, 1)

            n,m = self.lista[i].shape
            
            if (self.offsets[i] > 0):
                t = arange(self.offsets[i],n+self.offsets[i],1)
                axes = fig.add_subplot(311)
                axes.plot(t, x)
                axes.grid(True)
                axes = fig.add_subplot(312)
                axes.plot(t, y)
                axes.grid(True)
                axes = fig.add_subplot(313)
                axes.plot(t, z)
                axes.grid(True)
            else:
                axes = fig.add_subplot(311)
                axes.plot(x[-self.offsets[i]:])
                axes.grid(True)
                axes = fig.add_subplot(312)
                axes.plot(y[-self.offsets[i]:])
                axes.grid(True)
                axes = fig.add_subplot(313)
                axes.plot(z[-self.offsets[i]:])
                axes.grid(True)


 
    
if __name__ == "__main__":
    import doctest
    doctest.testmod()              
                    
