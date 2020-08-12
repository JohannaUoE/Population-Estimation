import numpy as np
from pandas import *
import math
import scipy.stats

'''
This script runs various statistical indeced on the calculated Parish counts 

Script by Johanna Kauffert
'''

def mea(true, pred,col):
    #mean absolute average error
    difference = abs(true-pred)
    print(f'{col}: Mean Absolute Error: {difference.mean()}')



def mape(true, pred,col):
    #mean absolute percentage error
    difference = []
    for i,j in zip(true,pred):
        difference.append(((abs(true-pred))/true)*100)
    #print(difference)
    difference = np.array(difference)
    print(f'{col}: Mean Absolute Percentage Error: {difference.mean()}')



def median_ea(true, pred,col):
    #median absolute average error
    difference = abs(true-pred)
    print(f'{col}: Median Absolute Error: {difference.median()}')


def rmse(true, pred,col):
    #root mean squared error
    difference = true-pred
    difference = difference**2
    summe = difference.mean()
    rmse = math.sqrt(summe)
    print(f'{col}: RMSE: {rmse}')


def rmse_perc(true, pred,col):
    #root mean squared error in percent
    difference = []
    for i,j in zip(true,pred):
        difference.append(((true-pred)/true)*100)

    difference = np.array(difference)

    difference = difference**2
    summe = difference.mean()
    rmse = math.sqrt(summe)
    print(f'{col}: % RMSE: {rmse}')


def pearson(true, pred,col):

    #pearson correlation coeeficient and r2 + Spearman
    true = np.array(true)
    plt.hist(true,bins = 30, color = "blue")
    plt.savefig("/exports/csce/datastore/geos/groups/MSCGIS/s1937352/ResultsJuly/Histograms/true.png")
    # test normality Test assumed normal or exponential 
    distribution using Lilliefors’ test. Lilliefors’ test is a Kolmogorov-Smirnov test with estimated parameters.
    norm = statsmodels.stats.diagnostic.kstest_normal(true, dist='norm', pvalmethod=None)
    print(f'Normal Ditribution True: {norm}')
    if norm[1] <= 0.05:
        print("Null Hypothesis rejected: No Normal Ditribution for True Values")


    pred = np.array(pred)
    norm = statsmodels.stats.diagnostic.kstest_normal(pred, dist='norm', pvalmethod=None)
    print(f'Normal Ditribution Pred ({col}): {norm}')
    if norm[1] <= 0.05:
        print("Null Hypothesis rejected: No Normal Ditribution for Predicted Values")

    plt.hist(pred,bins = 30, color = "blue")
    plt.savefig(f"/exports/csce/datastore/geos/groups/MSCGIS/s1937352/ResultsJuly/Histograms/pred_{col}.png")

    r,s = scipy.stats.pearsonr(true,pred)
    print(f'{col}: Pearson Correlation: {r}')
    print(f'{col}: Pearson Correlation Significane: {s}')
    r2 = r
    r2 = r2**2
    print(f'{col}: R2: {r2}')


    spear, sig = scipy.stats.spearmanr(true, pred, axis=0, nan_policy='propagate')
    print(f'{col}: Spearman Correlation: {spear}')
    print(f'{col}: Spearman Correlation Significane: {sig}')

def heteroscedasticity(true, pred,col):

    true = true.to_numpy()
    pred = pred.to_numpy()
    dif = pred -true
    # Calculate OLS
    x = sm.add_constant(true)
    model = sm.OLS(dif,x)
    results = model.fit()
    print(f'OLS Anlysis: {results.params}')

    #Calculate heteroscedasticity
    print(f'heteroscedasticity: {statsmodels.stats.diagnostic.het_breuschpagan(dif,x)}')

if __name__ == "__main__":
    data = read_csv("/exports/csce/datastore/geos/groups/MSCGIS/s1937352/DisFinal/ResultsRegressionAnalysis.csv", sep= ",")
    cols = []
    for col in data.columns: 
        cols.append(col)
    data = data.drop(data[data.True_sum == 0].index)
    true = data["True_sum"]
    for col in cols:
        if col.endswith("sum"):
            if col != "True_sum":
                print(f'*******************{col}*******************')
                pred = data[col]

                mea(true,pred,col)
                median_ea(true,pred,col)
                mape(true,pred,col)
                rmse(true, pred,col)
                rmse_perc(true, pred,col)
                pearson(true, pred,col)
                heteroscedasticity(true, pred,col)