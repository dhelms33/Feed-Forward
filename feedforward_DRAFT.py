from cgi import test
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class RandomDataGenerator:
    def __init__(self, n):
        self.n = n
    def generate(self):
        Xdata = np.random.rand(100, self.n)
        Ydata = 4+3*Xtest+np.random.randn(100,1)
        return Xdata, Ydata
        
#generate random data
class DataDefiner(RandomDataGenerator):
    def data_gettter():
        xtrain, ytrain, xtest, ytest = train_test_split(Xdata, Ydata, test_size=0.2)
        return xtrain, ytrain, xtest, ytest

class ModelMaker(DataDefiner):
    def fitter(self):
        self.model = FeedForwardNetwork()
        fitted = model.fit(xtrain, ytrain)
        return fitted
    