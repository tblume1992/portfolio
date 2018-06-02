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
    def fit(y, X, alpha, intercept = True):
    
        if intercept is True:
            constant = np.ones((X.shape[0],1))
            X = np.append(X,constant,1)
            
        I = np.eye(X.shape[1])
        ridge = inv(np.dot(X.T,X) + alpha*I)
        esti = np.dot(X.T,y)
        coefficients = pd.DataFrame(np.round(np.dot(ridge,esti),3))
        coefficients.columns = ['Coefficients']
        RidgeRegression.coefficients_ = coefficients
        RidgeRegression.fitted_ = np.dot(X,coefficients)
    def predict(X_test):
        constant = np.ones((X_test.shape[0],1))
        X_test = np.append(X_test,constant,1)
        RidgeRegression.predictions_ = np.dot(X_test,RidgeRegression.coefficients_)
