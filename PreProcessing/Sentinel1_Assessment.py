import numpy as np
from pandas import *
import math
import scipy.stats
from scipy.stats import linregress
import statsmodels
from statsmodels.stats import diagnostic
import matplotlib.pyplot as plt
import statsmodels.api as sm

'''
This script runs a correlation analysis based on Sentinel 1 VV Gamma0, VH Gamma0 and Interferometric Coherence of one scene
and population data

Script by Johanna Kauffert
'''

def pearson(true, intensity,col):

    #pearson correlation coeeficient and r2 + Spearman
    true = np.array(true)
    plt.hist(true,bins = 30, color = "blue")
    # test normality Test assumed normal or exponential distribution using Lilliefors’ test. Lilliefors’ test is a Kolmogorov-Smirnov test with estimated parameters.
    norm = statsmodels.stats.diagnostic.kstest_normal(true, dist='norm', pvalmethod=None)
    print(f'Normal Distribution True: {norm}')
    if norm[1] <= 0.05:
        print("Null Hypothesis rejected: No Normal Ditribution for True Values")


    intensity = np.array(intensity)
    norm = statsmodels.stats.diagnostic.kstest_normal(intensity, dist='norm', pvalmethod=None)
    print(f'Normal Distribution {col}: {norm}')
    if norm[1] <= 0.05:
        print("Null Hypothesis rejected: No Normal Ditribution for Predicted Values")

    r,s = scipy.stats.pearsonr(true,intensity)
    print(f'{col}: Pearson Correlation: {r}')
    print(f'{col}: Pearson Correlation Significacne: {s}')


    spear, sig = scipy.stats.spearmanr(true, intensity, axis=0, nan_policy='propagate')
    print(f'{col}: Spearman Correlation: {spear}')
    print(f'{col}: Spearman Correlation Significance: {sig}')


if __name__ == "__main__":
    data = read_csv("/exports/csce/datastore/geos/groups/MSCGIS/s1937352/DisFinal/Sentinel1_Assessment4.csv", sep= ",")
    cols = []
    for col in data.columns: 
        cols.append(col)
    pop = data["Pop_SexRat"]
    popkm = data["Pop_km"]
    for col in cols:
        if col.endswith("mean"):
            print(f'*******************{col}*******************')
            print(f'--------Mean-----------')
            intensity = data[col]
            print(f'+++++Population+++++')
            pearson(pop, intensity,col)
            print(f'+++++Population per km2+++++')
            pearson(popkm, intensity,col)

        if col.endswith("max"):
            intensity = data[col]
            print(f'--------Max-----------')
            print(f'+++++Population+++++')
            pearson(pop, intensity,col)
            print(f'+++++Population per km2+++++')
            pearson(popkm, intensity,col)
