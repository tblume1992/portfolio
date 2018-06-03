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
    def fit(y, X, intercept = True):
        if intercept is True:
            constant = np.ones((X.shape[0],1))
            X = np.append(X,constant,1)
        wls = inv(np.dot(X.T,X))
        esti = np.dot(X.T,y)
        coefficients = pd.DataFrame(np.round(np.dot(wls,esti),3))
        coefficients.columns = ['Coefficients']
        WLS.pre_wls = coefficients
        fitted_ = np.dot(X,coefficients)
        WLS.residuals_ = (y-fitted_)
        WLS.omega_ = np.diag(np.diag(np.cov(WLS.residuals_)))
        WLS.inv_omega = np.asarray(inv(WLS.omega_))
        
        wls = np.dot(X.T,WLS.inv_omega)
        wls = inv(np.dot(wls,X))
        esti = np.dot(X.T,WLS.inv_omega)
        esti = np.dot(esti,y)
        WLS.coefficients =pd.DataFrame(np.round(np.dot(wls,esti),3))
        variance = inv(np.dot(X.T,X))
        variance2 = np.dot(X.T,WLS.omega_)
        variance2 = np.dot(variance2,X)
        variance_final = np.dot(variance,variance2)
        WLS.variance_final = np.diag(np.dot(variance_final,variance))
        WLS.se_ = np.sqrt(WLS.variance_final)/np.sqrt(len(y))
        output = pd.DataFrame(WLS.coefficients)
        output.columns = ['Coefficients']
        output['Standard Error'] = WLS.se_
        output['T-Stat'] = output['Coefficients']/output['Standard Error']
        output['p-values'] = stats.t.sf(np.abs(output['T-Stat']), len(y)-1)*2.0
        output = output.round(decimals = 3)
        print("****************************WLS Results****************************")
        print(output)
    def predict(X_test):
        constant = np.ones((X_test.shape[0],1))
        X_test = np.append(X_test,constant,1)
        WLS.predictions_ = np.dot(X_test,WLS.coefficients_)