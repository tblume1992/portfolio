#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 21:40:58 2018

@author: tyler
"""

from numpy.linalg import inv
import pandas as pd
import numpy as np
from scipy import stats

class WLS():
    def __init__(self, intercept = True):
        self.intercept = intercept
    def add_constant(self, X):
        constant = np.ones((X.shape[0],1))
        return np.append(X,constant,1)
    def ols_fit(self,y, X):
        if self.intercept:
            X = self.add_constant(X)
        ols = inv(np.dot(X.T,X))
        esti = np.dot(X.T,y)
        coefficients = pd.DataFrame(np.round(np.dot(ols,esti),3))
        coefficients.columns = ['Coefficients']
        return np.dot(X,coefficients)
    def get_weight(self):
        omega = np.diag(np.diag(np.cov(self.residuals_)))
        return np.asarray(inv(omega))
    def fit(self, y, X):
        self.residuals_ = (y-self.ols_fit(y,X))  
        if self.intercept:    
            X = self.add_constant(X)
        wls = np.dot(X.T,self.get_weight())
        wls = inv(np.dot(wls,X))
        esti = np.dot(X.T,self.get_weight())
        esti = np.dot(esti,y)
        self.coefficients_ =pd.DataFrame(np.round(np.dot(wls,esti),3))
        variance = inv(np.dot(X.T,X))
        variance2 = np.dot(X.T,self.get_weight())
        variance2 = np.dot(variance2,X)
        variance_final = np.dot(variance,variance2)
        self.variance_final = np.diag(np.dot(variance_final,variance))
        self.se_ = np.sqrt(self.variance_final)/np.sqrt(len(y))
        output = pd.DataFrame(self.coefficients_)
        output.columns = ['Coefficients']
        output['Standard Error'] = self.se_
        output['T-Stat'] = output['Coefficients']/output['Standard Error']
        output['p-values'] = stats.t.sf(np.abs(output['T-Stat']), len(y)-1)*2.0
        output = output.round(decimals = 3)
        print("****************************WLS Results****************************")
        print(output)
    def predict(self, X_test):
        if self.intercept:
            X_test = self.add_constant(X_test)
        return np.dot(X_test,self.coefficients_)
