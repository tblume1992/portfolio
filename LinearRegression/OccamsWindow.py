# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 15:21:04 2018
@author: t-blu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 14:44:06 2018
@author: tyler
"""
import numpy as np
import pandas as pd
import itertools
import statsmodels.api as sm
from scipy import stats
from tqdm import tqdm
def OccamsWindow(y, X , window= 6):
    X = pd.DataFrame(X)
    mod = sm.GLM(y, X)
    sat = mod.fit()
    saturated_model = sm.regression.linear_model.RegressionResults.bic(sat)
    print('Saturated model BIC:' + str(saturated_model))
    print('****************************Saturated Model Results****************************')
    print(sat.summary())
    bic_dataset = []
    coefficient_dataset = []
    se = []
    weighted_coefficients = []
    weighted_se = []    
    for L in tqdm(range(0, len(X.columns.values)+1)):
        for subset in itertools.combinations(X.columns.values, L):
            if len(subset) < 1:
                pass
            else:
                mod = sm.GLM(y, X.get(list(subset)))
                res = mod.fit()
                bic = sm.regression.linear_model.RegressionResults.bic(res)
                if bic <= saturated_model - window:
                    coefficient_dataset.append(res.params)
                    se.append(res.bse)
                    bic_dataset.append(bic)
    if not bic_dataset:
        print('No models found in Window.  Please try a different window size!')
    else:   
        bic_average = np.average(bic_dataset)
        bic_dataset = np.square(np.divide(1,bic_dataset)*np.max(bic_dataset))
        for i in range(len(bic_dataset)):
            weighted_coefficients.append((coefficient_dataset[i] * bic_dataset[i]) / sum(bic_dataset))
            weighted_se.append((se[i] * bic_dataset[i]) / sum(bic_dataset))
        weighted_coefficients = pd.Series(weighted_coefficients).apply(pd.Series) 
        weighted_se = pd.Series(weighted_se).apply(pd.Series) 
        standard_errors = pd.DataFrame(weighted_se.sum(axis = 0))
        coefficients = pd.DataFrame(weighted_coefficients.sum(axis = 0))
        output = coefficients.join(standard_errors,lsuffix='Coefficients', rsuffix='Standard Error')
        output.columns = ['Coefficients','Standard Error']
        output['T-Stat'] = output['Coefficients']/output['Standard Error']
        output['p-values'] = stats.t.sf(np.abs(output['T-Stat']), len(y)-1)*2.0
        output = output.round(decimals = 3)
        print("****************************Occam's Window Results****************************")
        print("Window Size: " + str(len(bic_dataset)))
        print("Average BIC in Window: " + str(bic_average))
        print(output)
    return bic_dataset, weighted_coefficients, weighted_se
