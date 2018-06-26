#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 17:43:42 2018

@author: tyler
"""
from numpy.linalg import inv
import pandas as pd
import numpy as np

class RidgeRegression():
    def __init__(self, intercept = True, alpha = .01):
        self.intercept = intercept
        self.alpha = alpha
    def add_constant(self, X):
        constant = np.ones((X.shape[0],1))
        return np.append(X,constant,1)
    def fit(self, y, X):
        if self.intercept:
            X = self.add_constant(X)
        I = np.eye(X.shape[1])
        ridge = inv(np.dot(X.T,X) + self.alpha*I)
        esti = np.dot(X.T,y)
        coefficients = pd.DataFrame(np.round(np.dot(ridge,esti),3))
        self.coefficients_ = coefficients
        self.fitted_ = np.dot(X,coefficients)
    def predict(self,X_test):
        if self.intercept:
            X_test = self.add_constant(X_test)
        self.predictions_ = np.dot(X_test,self.coefficients_)
