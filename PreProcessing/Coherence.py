import os
import subprocess

''' 
This Script is a wrapper to get the coherence for two Sentinel 2 SLC
scenes. The used Software is Gamma
The commands were utilitised and slightly changed from a gamma coherence Demo

Script by Johanna Kauffert
'''

#specify paths to DEM
dem = "/exports/csce/datastore/geos/groups/MSCGIS/s1937352/DEM_Uganda/UgandaB.dem"
dem_par = "/exports/csce/datastore/geos/groups/MSCGIS/s1937352/DEM_Uganda/UgandaB.dem_par"


#specify rootdirectory

#rootdir = '/exports/csce/datastore/geos/groups/geos_EO/S1937352/Scene'
rootdir = '/exports/csce/datastore/geos/groups/MSCGIS/s1937352/Sentinel1_Coherence/SentinelSLC'
counter = 1
outdir = '/exports/csce/datastore/geos/groups/MSCGIS/s1937352/Sentinel1_Coherence/CoherenceTiffs'

# all gamma commands are put together as strings and os.system prints the 
# statement into bash and executes it.

for dirname in os.listdir(rootdir):
    
    
    #specify filepaths
    dir = f'{rootdir}/{dirname}'
    print(dir)

    ### beginning with the reference scene which is always the earlier one
    ########################################################
    # 1) For the reference S1 IWS scene


    # go into the early directory and get the date of the file as well as the name of the zipfile
    for efile in os.listdir(f'{dir}/Early/'):
        earlyfilezip = efile
        print(earlyfilezip)
    
    earlyfile = earlyfilezip[:-4]
    earlydate = earlyfile[17:-42]
    mkdir = f"mkdir {dir}/temp"
    os.system(mkdir)
    diros = f'{dir}/temp'
    os.chdir(diros)
    
    
    ###S1_BURST_tab_from_zipfile: Script used to generate S1_BURST_tab to support burst selection###
    s1_burst = f"S1_BURST_tab_from_zipfile - {dir}/Early/{earlyfilezip}"
    os.system(s1_burst)
    
    
    makeziplist = f"ls {dir}/Early/*{earlydate}*.zip > {dir}/zipfile_list_{earlydate}"
    os.system(makeziplist)
    
    ###S1_import_SLC_from_zipfiles: Script to read in and concatenate S1 TOPS SLC from zip files###
    # all swaths and bursts are imported, this can be changed
    s1_import = f"S1_import_SLC_from_zipfiles {dir}/zipfile_list_{earlydate} - vv 1 0 . 1 1"
    os.system(s1_import)
    
    #write all file names into a tab file
    #I think this is not needed anymore
    #echotask = f'echo "{earlydate}.vv.mli.iw1 {earlydate}.vv.mli.iw1.par {earlydate}.vv.mli.iw1.tops_par \n{earlydate}.vv.mli.iw2 {earlydate}.vv.mli.iw2.par {earlydate}.vv.mli.iw2.tops_par \n{earlydate}.vv.mli.iw3 {earlydate}.vv.mli.iw3.par {earlydate}.vv.mli.iw3.tops_par" > {earlydate}.vv.MLI_tab'
    #os.system(echotask)
    

    ###Calculate MLI mosaic from ScanSAR SLC burst data (Sentinel-1, TerraSAR-X, RCM...)###
    multi_look_ScanSAR = f'multi_look_ScanSAR {earlydate}.vv.SLC_tab {earlydate}.vv.mli {earlydate}.vv.mli.par 20 4 1'
    os.system(multi_look_ScanSAR)
    
    # move on to the other file 

    for lfile in os.listdir(f'{dir}/Later/'):
        laterfilezip = lfile
        print(laterfilezip)
    laterfile = laterfilezip[:-4]
    laterdate = laterfile[17:-42]
    
    ###S1_BURST_tab_from_zipfile: Script used to generate S1_BURST_tab to support burst selection###
    s1_burst = f"S1_BURST_tab_from_zipfile - {dir}/Later/{laterfilezip}"
    os.system(s1_burst)

    makeziplist2 = f"ls {dir}/Later/*{laterdate}*.zip > {dir}/zipfile_list_{laterdate}"
    os.system(makeziplist2)

    ###S1_import_SLC_from_zipfiles: Script to read in and concatenate S1 TOPS SLC from zip files###
    # all swaths and bursts are imported, this can be changed
    s1_import = f"S1_import_SLC_from_zipfiles {dir}/zipfile_list_{laterdate} - vv 1 0 . 1 1"
    os.system(s1_import)
    

    ###Calculate MLI mosaic from ScanSAR SLC burst data (Sentinel-1, TerraSAR-X, RCM...)###
    multi_look_ScanSAR2 = f'multi_look_ScanSAR {laterdate}.vv.SLC_tab {laterdate}.vv.mli {laterdate}.vv.mli.par 20 4 1'
    os.system(multi_look_ScanSAR2)
    
    #make RLSC_tab for later date
    sed = f"sed s/.slc/.rslc/g < {laterdate}.vv.SLC_tab > {laterdate}.vv.RSLC_tab"
    os.system(sed)

    ###S1_coreg_TOPS: Script to coregister a Sentinel-1 TOPS mode burst SLC to a reference burst SLC###
    #this step does a shit of things and calculates ages :-D
    S1_coreg_TOPS = f'S1_coreg_TOPS {earlydate}.vv.SLC_tab {earlydate}.vv {laterdate}.vv.SLC_tab {laterdate}.vv {laterdate}.vv.RSLC_tab 5.0 10 2'
    os.system(S1_coreg_TOPS)

    
    mli_samples = subprocess.check_output(f"grep samples {earlydate}.vv.rmli.par", shell=True)
    mli_samples = str(mli_samples).replace("\n'","").split(' ')[-1][:-3]
    print("MLI Samples:", mli_samples)
    
    ######### Calculate Coherence (2Types)#############

    ###Estimate interferometric coherence####
    cc_wave = f"cc_wave {earlydate}.vv_{laterdate}.vv.diff {earlydate}.vv.rmli {laterdate}.vv.mli {earlydate}_{laterdate}.vv.diff.cc {mli_samples} 5. 5. 0"
    os.system(cc_wave)

    ###LAT cc_ad: Adaptive coherence estimation with consideration of phase slope and texture####
    cc_ad = f"cc_ad {earlydate}.vv_{laterdate}.vv.diff {earlydate}.vv.rmli {laterdate}.vv.mli - - {earlydate}_{laterdate}.vv.diff.cc_ad {mli_samples} 3. 9. 0"
    os.system(cc_ad)
    

    ######## The next steps are for geocoding the coherence image. This mist be performed by calculating a look up table ##########
    gc_map = f"gc_map2 {earlydate}.vv.rmli.par {dem_par} {dem} EQA.dem_par EQA.dem {earlydate}.lt 1. 1. {earlydate}.ls_map - {earlydate}.inc"
    os.system(gc_map)
    
    
    #gc_map2
    ###Calculate terrain-based sigma0 and gammma0 normalization area in slant-range geometry####
    pixel_area = f"pixel_area {earlydate}.vv.rmli.par EQA.dem_par EQA.dem {earlydate}.lt {earlydate}.ls_map {earlydate}.inc {earlydate}.pix_sigma0 {earlydate}.pix_gamma0"
    os.system(pixel_area)
    

    create_diff_par = f"create_diff_par {earlydate}.vv.rmli.par - {earlydate}.diff_par 1 0"
    os.system(create_diff_par)
    
    offset_pwrm = f"offset_pwrm {earlydate}.pix_gamma0 {earlydate}.vv.rmli {earlydate}.diff_par {earlydate}.offs {earlydate}.snr 128 128 offsets 3 24 24 0.15"
    os.system(offset_pwrm)

    offset_fitm = f"offset_fitm {earlydate}.offs {earlydate}.snr {earlydate}.diff_par {earlydate}.coffs {earlydate}.coffsets 0.2 1"
    os.system(offset_fitm)
    
    # get the DEM width 
    dem_samples = subprocess.check_output(f"grep width EQA.dem_par", shell=True) 
    dem_samples = str(dem_samples).replace("\n'","").split(' ')[-1][:-3]
    print("DEM Smaples:", dem_samples)

    ## Geocoding lookup table refinement using DIFF_par offset polynomials ##
    gc_map_fine = f"gc_map_fine {earlydate}.lt {dem_samples} {earlydate}.diff_par {earlydate}.lt_fine 1"
    os.system(gc_map_fine)
    
    ## Calculate terrain-based sigma0 and gammma0 normalization area in slant-range geometry ##
    pixel_area2 = f"pixel_area {earlydate}.vv.rmli.par EQA.dem_par EQA.dem {earlydate}.lt_fine {earlydate}.ls_map {earlydate}.inc {earlydate}.pix_sigma0 {earlydate}.pix_gamma0"
    os.system(pixel_area2)
    


    #get the width of look up table
    lookupwidth = subprocess.check_output(f"grep width EQA.dem_par", shell=True)
    lookupwidth = str(lookupwidth).replace("\n'","").split(' ')[-1][:-3]
    print("Look Up Table Width:", lookupwidth)


    #get the lines of look up table
    lookupline = subprocess.check_output(f"grep line EQA.dem_par", shell=True)
    lookupline = str(lookupline).replace("\n'","").split(' ')[-1][:-3]
    print("Look Up Table Lines:", lookupline)
    

    ###Geocoding of image data using a geocoding lookup table###
    ## Coherence ###
    geocode_back = f"geocode_back {earlydate}_{laterdate}.vv.diff.cc_ad {mli_samples} {earlydate}.lt_fine EQA_{earlydate}_{laterdate}.vv.diff.cc_ad {lookupwidth} {lookupline} 5 0"
    os.system(geocode_back)


    ####Computing Output (2) ######

    # Output is only coherence  #
    
    data2geotiff1 = f"data2geotiff EQA.dem_par EQA_{earlydate}_{laterdate}.vv.diff.cc_ad 2 {outdir}/{earlydate}_{laterdate}_{counter}_coherence.tif -99" 
    os.system(data2geotiff1)
    
    delete = f"rm -rf {dir}/temp"
    os.system(delete)
    

    counter +=1



