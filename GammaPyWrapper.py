
#import necessary modules
import os
import subprocess


'''
This script processes Sentinel1 GRD imgages to VV and VH gamma0 backscatter with Gamma RS
Specify the file paths as well as a DEM. S1.zip files must be unzipped to .SAFE folders and the
correct orbit file must be saved insight the .SAFE file (/osv/POEORB/)

The gamma script was put together by John Truckenbrodt 
(Truckenbrodt, John, et al. "Towards Sentinel-1 SAR analysis-ready data: A best practices assessment
 on preparing backscatter data for the cube." Data 4.3 (2019): 93)

'''

#set the file paths
dem = "/home/s1937352/Dissertation_Uganda/DisApril/Uganda_4326.dem"
dem_par = "/home/s1937352/Dissertation_Uganda/DisApril/Uganda_4326.dem_par"
outdir = "/home/s1937352/Dissertation_Uganda/GRDOutput"

rootdir = '/home/s1937352/GRDH/'

def processGamma():
    #loop over all files in the specified folder
    for dirname in os.listdir(rootdir):
        #specify filepaths
        dir = f'/home/s1937352/GRDH/{dirname}'
        print(dirname)
        filename= str(dirname).lower().replace("_", "-")[:-5]
        print(filename)
        filenameVH = filename.replace("1sdv","vh").replace("grdh","grd")
        filenameVV = filename.replace("1sdv","vv").replace("grdh","grd")
        # PAR_S1_GRD
        par_command1= f"par_S1_GRD {dir}/measurement/{filenameVH}-002.tiff {dir}/annotation/{filenameVH}-002.xml {dir}/annotation/calibration/calibration-{filenameVH}-002.xml - {dir}/{filenameVH}_VH_grd.par {dir}/{filenameVH}_VH_grd - - - - -"
        os.system(par_command1)
        par_command2= f"par_S1_GRD {dir}/measurement/{filenameVV}-001.tiff {dir}/annotation/{filenameVV}-001.xml {dir}/annotation/calibration/calibration-{filenameVV}-001.xml - {dir}/{filenameVV}_VV_grd.par {dir}/{filenameVV}_VV_grd - - - - -"
        os.system(par_command2)

        # Apply Orbit files
        # correct orb files must be allocated in SAFE folder 
        for file in os.listdir(f'{dir}/osv/POEORB/'):
            if file.endswith("EOF"):
                orb = file
        opod1 = f"S1_OPOD_vec {dir}/{filenameVH}_VH_grd.par {dir}/osv/POEORB/{orb} -"
        opod2 = f"S1_OPOD_vec {dir}/{filenameVV}_VV_grd.par {dir}/osv/POEORB/{orb} -"
        os.system(opod1)
        os.system(opod2)

        #Multilook image
        multlook1 = f"multi_look_MLI {dir}/{filenameVH}_VH_grd {dir}/{filenameVH}_VH_grd.par {dir}/{filenameVH}_VH_grd_mli {dir}/{filenameVH}_VH_grd_mli.par 2 2 - - -"
        multlook2 = f"multi_look_MLI {dir}/{filenameVV}_VV_grd {dir}/{filenameVV}_VV_grd.par {dir}/{filenameVV}_VV_grd_mli {dir}/{filenameVV}_VV_grd_mli.par 2 2 - - -"
        os.system(multlook1)
        os.system(multlook2)

        #Calculate terrain-geocoding lookup table and DEM derived data products
        gc_map = f"gc_map {dir}/{filenameVH}_VH_grd_mli.par - {dem_par} {dem} {dir}/{filename}_dem_seg_geo.par {dir}/{filename}_dem_seg_geo {dir}/{filename}_lut_init 1.0 1.0 - - - {dir}/{filename}_inc_geo - {dir}/{filename}_pix_geo {dir}/{filename}_ls_map_geo 8 2 -"
        os.system(gc_map)

        #Calculate terrain-based sigma0 and gammma0 normalization area in slant-range geometry
        pixel_area = f"pixel_area {dir}/{filenameVH}_VH_grd_mli.par {dir}/{filename}_dem_seg_geo.par {dir}/{filename}_dem_seg_geo {dir}/{filename}_lut_init {dir}/{filename}_ls_map_geo {dir}/{filename}_inc_geo - - - - {dir}/{filename}_pix_fine -"
        os.system(pixel_area)

        #Calculate product of two images: (image 1)*(image 2) ----  VH polarisation
        mli_samples = subprocess.check_output(f"grep samples {dir}/{filenameVH}_VH_grd_mli.par", shell=True)
        mli_samples = str(mli_samples).replace("\n'","").split(' ')[-1][:-3]
        print("MLI Samples:", mli_samples)
        product1 = f"product {dir}/{filenameVH}_VH_grd_mli {dir}/{filename}_pix_fine {dir}/{filenameVH}_VH_grd_mli_pan {mli_samples} 1 1 -"
        os.system(product1)

        #Geocoding of image data using a geocoding lookup table ----  VH polarisation
        dem_samples = subprocess.check_output(f"grep width {dir}/{filename}_dem_seg_geo.par", shell=True)
        dem_samples = str(dem_samples).replace("\n'","").split(' ')[-1][:-3]
        print("DEM Smaples:", dem_samples)
        geocode_back1 = f"geocode_back {dir}/{filenameVH}_VH_grd_mli_pan {mli_samples} {dir}/{filename}_lut_init {dir}/{filenameVH}_VH_grd_mli_pan_geo {dem_samples} - 2 - - - -"
        os.system(geocode_back1)

        #Compute backscatter coefficient gamma (sigma0)/cos(inc) ----  VH polarisation
        sigma2gamma1 = f"sigma2gamma {dir}/{filenameVH}_VH_grd_mli_pan_geo {dir}/{filename}_inc_geo {dir}/{filenameVH}_VH_grd_mli_norm_geo {dem_samples}"
        os.system(sigma2gamma1)

        #Calculate product of two images: (image 1)*(image 2) ----  VV polarisation
        product2 = f"product {dir}/{filenameVV}_VV_grd_mli {dir}/{filename}_pix_fine {dir}/{filenameVV}_VV_grd_mli_pan {mli_samples} 1 1 -"
        os.system(product2)

        #Geocoding of image data using a geocoding lookup table ----  VV polarisation
        geocode_back2 = f"geocode_back  {dir}/{filenameVV}_VV_grd_mli_pan {mli_samples} {dir}/{filename}_lut_init {dir}/{filenameVV}_VV_grd_mli_pan_geo {dem_samples} - 2 - - - -"
        os.system(geocode_back2) 

        #Compute backscatter coefficient gamma (sigma0)/cos(inc) ----  VHVpolarisation
        sigma2gamma2 = f"sigma2gamma {dir}/{filenameVV}_VV_grd_mli_pan_geo {dir}/{filename}_inc_geo {dir}/{filenameVV}_VV_grd_mli_norm_geo {dem_samples}"
        os.system(sigma2gamma2) 

        #Conversion of data between linear and dB scale
        linear_to_dB1 = f"linear_to_dB {dir}/{filenameVH}_VH_grd_mli_norm_geo {dir}/{filenameVH}_VH_grd_mli_norm_geo_db {dem_samples} 0 -99"
        os.system(linear_to_dB1) 
        linear_to_dB2 = f"linear_to_dB {dir}/{filenameVV}_VV_grd_mli_norm_geo {dir}/{filenameVV}_VV_grd_mli_norm_geo_db {dem_samples} 0 -99"
        os.system(linear_to_dB2)

        # Write Outout as Geotiff
        data2geotiff1 = f"data2geotiff {dir}/{filename}_dem_seg_geo.par {dir}/{filenameVH}_VH_grd_mli_norm_geo_db 2 {outdir}/{filenameVH}_VH_grd_mli_norm_geo_db.tif -99" 
        data2geotiff2 = f"data2geotiff {dir}/{filename}_dem_seg_geo.par {dir}/{filenameVV}_VV_grd_mli_norm_geo_db 2 {outdir}/{filenameVV}_VV_grd_mli_norm_geo_db.tif -99" 
        os.system(data2geotiff1)
        os.system(data2geotiff2)


        print("I finished the scene")


processGamma()