import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import matplotlib as mpl
from matplotlib import cm
from scipy.interpolate import interpn
from matplotlib.colors import Normalize


'''
This script runs produces density scatter plots of Sentinel 1 VV Gamma0, VH Gamma0 and Interferometric Coherence of one scene
and population data

Script by Johanna Kauffert
'''


def density_scatter( x , y, ax = None, ylabel="ylabel", sort = True, bins = 20, **kwargs )   :
    ## Function to produce a density scatter plot ####

    ### This function is partyl taken from:
    ###https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib####

    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)


    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    plt.scatter( x, y, c=z, s= 2, **kwargs )
    plt.xlabel("Population Counts")
    plt.ylabel(ylabel)

    norm = Normalize(vmin = np.min(z), vmax = np.max(z))

    plt.savefig(f'/exports/csce/datastore/geos/groups/MSCGIS/s1937352/DisFinal/{ylabel}.png')
    plt.clf()



if "__main__" == __name__ :

    #set directory
    csvdir = "/exports/csce/datastore/geos/groups/MSCGIS/s1937352/DisFinal/"

    #specify csv for usage
    data = pd.read_csv(f"{csvdir}Sentinel1_Assessment4.csv", sep = "," )

    #read columns to variables
    pop = data["Pop_SexRat"]
    coh_mean = data["Coh_mean"]
    coh_max = data["Coh_max"]
    VV_mean = data["VV_mean"]
    VV_max = data["VV_max"]
    VH_mean = data["VH_mean"]
    VH_max = data["VH_max"]

    # turn pf data columns to numpy arrays
    pop = pop.to_numpy()
    coh_mean  = coh_mean.to_numpy()
    coh_max  = coh_max.to_numpy()
    VV_mean  = VV_mean.to_numpy()
    VV_max  = VV_max.to_numpy()
    VH_mean  = VH_mean.to_numpy()
    VH_max  = VH_max.to_numpy()
    liste = ['coh_mean','coh_max','VV_mean','VV_max','VH_mean','VH_max']

    #call denisty scatter function for all columns
    density_scatter(pop , coh_mean, ylabel = "Mean Coherence per Parish", bins = [30,30] )
    density_scatter(pop , coh_max, ylabel = "Max Coherence per Parish", bins = [30,30] )
    density_scatter(pop , VV_mean, ylabel = "Mean VV Gamma0 per Parish",bins = [30,30] )
    density_scatter(pop , VV_max, ylabel = "Max VV Gamma0 per Parish", bins = [30,30] )
    density_scatter(pop , VH_mean,ylabel = "Mean VH Gamma0 per Parish", bins = [30,30] )
    density_scatter(pop , VH_max,ylabel = "Max VH Gamma0 per Parish", bins = [30,30] )