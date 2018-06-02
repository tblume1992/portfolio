#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 08:56:36 2018

@author: tyler
"""
from numpy.linalg import inv
import pandas as pd
import numpy as np

class OLS():
    def fit(y, X, intercept = True):
    
        if intercept is True:
            constant = np.ones((X.shape[0],1))
            X = np.append(X,constant,1)
        ols = inv(np.dot(X.T,X))
        esti = np.dot(X.T,y)
        coefficients = pd.DataFrame(np.round(np.dot(ols,esti),3))
        coefficients.columns = ['Coefficients']
        OLS.coefficients_ = coefficients
        OLS.fitted_ = np.dot(X,coefficients)
    def predict(X_test):
        constant = np.ones((X_test.shape[0],1))
        X_test = np.append(X_test,constant,1)
        OLS.predictions_ = np.dot(X_test,OLS.coefficients_)
