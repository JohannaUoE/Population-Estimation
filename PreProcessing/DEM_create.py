'''
    This script downloads a SRTM DEM to the extent of a specified shapfiles with PyroSAR package.
    With dem_import command teh tif is transferred to a Gamma DEM file

    Script by Johanna Kauffert (based on John Truckenbrodt 2018)
'''

#import nescessary packages
from pyroSAR.auxdata import dem_autoload, dem_create
import re
import os
from spatialist import gdalwarp, Vector

#set directories
maindir = '/exports/csce/datastore/geos/groups/MSCGIS/s1937352/DEM_Uganda'
os.chdir(maindir)

#specify shapefile of region of interest
roi = '/exports/csce/datastore/geos/groups/MSCGIS/s1937352/DEM_Uganda/UgandaBigBox.shp'


epsg=21036 #specify epsg
demType = 'SRTM 1Sec HGT' #specify dem type (Others: AW3D30, SRTM 3Sec)
dem_name = 'UgandaBig10'
dem_path = f'{maindir}/{dem_name}/.tif'
if not os.path.isfile(dem_path):
    #create VRT with the specified DEM and Bounding Box
    with Vector(roi) as bbox:
        vrt = dem_autoload(geometries=[bbox], 
                           demType=demType,
                           vrt=f'/{dem_name}.vrt',
                           buffer=0.1)
    # create a tif from the VRT with a given resolution
    dem_create(src=vrt, dst=dem_path, t_srs=epsg, tr=(10, 10), geoid_convert=True)

# convert tif to Gamma files (dem, dem_par) 
gammastring = f'dem_import UgandaBig10.tif UgandaB.dem UgandaB.dem_par - - /opt/gamma/20190613/DIFF/scripts/egm96.dem /opt/gamma/20190613/DIFF/scripts/egm96.dem_par'
os.system(gammastring)