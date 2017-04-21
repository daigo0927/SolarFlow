# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pdb

from PIL import Image
from scipy.stats import multivariate_normal

def sigmoid(z):
    return 1/(1+np.exp(-z))

class Norm2Dmix:

    def __init__(self, mus, covs, pi):
        self.mus = mus
        self.covs = covs
        self.pi = pi

        self.Norms = [multivariate_normal(mean = mu, cov = cov) \
                      for mu, cov in zip(self.mus, self.covs)]

    def pdf(self, x):
        q = np.array([Norm.pdf(x = x) \
                      for Norm in self.Norms])
        q_WeightedSum = np.sum(q * self.pi)
        
        return q_WeightedSum

    def pdf_each(self, x):
        q = np.array([Norm.pdf(x = x) \
                      for Norm in self.Norms])
        return q

class Epanechnikov2Dfunc:

    ## return without 0-cut value
    # 2 variate Epanechnikov functions
    # f(x) = max(0, D-(x-mean)*cov*(x-mean.T))
    # D : Normalize constant
        
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
        
        h = 1e-4
        a = cov[0,0]
        b = cov[0,1]
        c = cov[1,1]
        if(a == c):
            c = c+h
            
        theta = 1/2*np.arctan(2*b/(a-c))
        tmp1 = \
               c*np.cos(theta)**2 - 2*b*np.sin(theta)*np.cos(theta) + a*np.sin(theta)**2
        tmp2 = \
               c*np.sin(theta)**2 + 2*b*np.sin(theta)*np.cos(theta) + a*np.cos(theta)**2
        det = a*c-b**2
        self.D = np.sqrt(2 * np.sqrt(tmp1*tmp2)/np.pi/det)
            
    def value(self, x):
        mean = self.mean
        cov = self.cov
        D = self.D
        
        value = D - np.dot((x-mean), np.linalg.solve(cov, (x-mean).T))
        return value

class Epanechnikov2D:

    # 2 variate Epanechnikov function
    # f(x) = max(0, Epa2Dfunc)
    # D : normalize constant

    def __init__(self, mean, cov):
        self.Epa = Epanechnikov2Dfunc(mean = mean, cov = cov)

    def pdf(self, x):
        density = max(0, self.Epa.value(x = x))

        return density
    
class Epanechnikov2Dmix:

    def __init__(self, mus, covs, pi):
        self.mus = mus
        self.covs = covs
        self.pi = pi

        self.Epas = [Epanechnikov2D(mean = mu, cov = cov) \
                     for mu, cov in zip(self.mus, self.covs)]

        self.p_each = None

    def pdf_each(self, x):
        self.p_each = np.array([Epa.pdf(x = x) \
                                for Epa in self.Epas])
        return self.p_each

    def pdf(self, x):
        _ = self.pdf_each(x = x)

        pdf_WeightedSum = np.sum(self.p_each * self.pi)

        return pdf_WeightedSum
    
            
    
    
