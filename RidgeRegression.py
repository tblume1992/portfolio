#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 17:43:42 2018

@author: tyler
"""
from numpy.linalg import inv
import pandas as pd
import numpy as np

def Ridge(y, X, alpha, intercept = True):
    if intercept is True:
        intercept = np.ones(5)
    I = np.eye(X.shape[1])
    ridge = inv(np.dot(X.T,X) + alpha*I)
    esti = np.dot(X.T,y)
    coefficients = pd.DataFrame(np.round(np.dot(ridge,esti),3))
    coefficients.columns = ['Coefficients']
    Ridge.coefficients = coefficients
    Ridge.fitted = np.dot(X,coefficients)

