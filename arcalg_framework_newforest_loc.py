import matplotlib.pyplot as plt
import pyart
import numpy as np
import numpy.ma as ma
from metpy.units import check_units, concatenate, units
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from siphon.radarserver import RadarServer
#rs = RadarServer('http://thredds-aws.unidata.ucar.edu/thredds/radarServer/nexrad/level2/S3/')
#rs = RadarServer('http://thredds.ucar.edu/thredds/radarServer/nexrad/level2/IDD/')
from datetime import datetime, timedelta
from siphon.cdmr import Dataset
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from metpy.units import check_units, concatenate, units
from shapely.geometry import polygon as sp
import pyproj 
import shapely.ops as ops
from shapely.ops import transform
from shapely.geometry.polygon import Polygon
from functools import partial
from shapely import geometry
import netCDF4
from scipy import ndimage as ndi
#from skimage.feature import peak_local_max
#from skimage import data, img_as_float
from pyproj import Geod
from metpy.calc import wind_direction, wind_speed, wind_components
import matplotlib.lines as mlines
import pandas as pd
import scipy.stats as stats
import csv
import pickle
from sklearn.ensemble import RandomForestClassifier
import nexradaws
import os
from grid_section_arcalg import gridding_arcalg
from kdp_section import kdp_genesis
from gradient_section_arcalg import grad_mask_arcalg
from ungridded_arcalg import quality_control_arcalg
from stormid_section import storm_objects
from zdr_arc_section import zdrarc
#from hail_section import hail_objects
from zhh_section import zhh_objects
from kdpfoot_section import kdp_objects
#from zdr_col_section import zdrcol

def multi_case_algorithm_ML1_newforest_loc(storm_relative_dir, zdrlev, kdplev, REFlev, REFlev1, big_storm, zero_z_trigger, storm_to_track, year, month, day, hour, start_min, duration, calibration, station, localfolder, Bunkers_m, track_dis=10, GR_mins=1.0):
    #Set vector perpendicular to FFD Z gradient
    storm_relative_dir = storm_relative_dir
    #Set storm motion
    Bunkers_m = Bunkers_m
    #Set ZDR Threshold for outlining arcs
    zdrlev = [zdrlev]
    #Set KDP Threshold for finding KDP feet
    kdplev = [kdplev]
    #Set reflectivity thresholds for storm tracking algorithm
    REFlev = [REFlev]
    REFlev1 = [REFlev1]
    #Set storm size threshold that triggers subdivision of big storms
    big_storm = big_storm #km^2
    Outer_r = 30 #km
    Inner_r = 6 #km
    #Set trigger to ignore strangely-formatted files right before 00Z
    #Pre-SAILS #: 17
    #SAILS #: 25
    zero_z_trigger = zero_z_trigger
    storm_to_track = storm_to_track
    zdr_outlines = []
    #Here, set the initial time of the archived radar loop you want.
    #Our specified time
    dt = datetime(year,month, day, hour, start_min)
    station = station
    end_dt = dt + timedelta(hours=duration)
    folder=localfolder
    print(dt, end_dt)
    #Set up nexrad interface
    #conn = nexradaws.NexradAwsInterface()
    #scans = conn.get_avail_scans_in_range(dt,end_dt,station)
    #results = conn.download(scans, 'RadarFolder')

    #Setting counters for figures and Pandas indices
    f = 27
    n = 1
    storm_index = 0
    scan_index = 0
    tracking_index = 0
    #Create geod object for later distance and area calculations
    g = Geod(ellps='sphere')
    #Open the placefile
    f = open("ARCALG_newforest"+station+str(dt.year)+str(dt.month)+str(dt.day)+str(dt.hour)+str(dt.minute)+"_Placefile.txt", "w+")
    f.write("Title: ZDR Arc Algorithm Placefile \n")
    f.write("Refresh: 8 \n \n")

    #Load ML algorithm
    forest_loaded = pickle.load(open('NewData2022RandomForest.pkl', 'rb'))
#    forest_loaded_col = pickle.load(open('BestRandomForestColumnsLEN200.pkl', 'rb'))

    #Actual algorithm code starts here
    #Create a list for the lists of arc outlines
    zdr_out_list = []
    tracks_dataframe = []
    #for i,scan in enumerate(results.iter_success(),start=1):
    #Local file option:
    for radar_file in os.listdir(folder):
        #Loop over all files in the dataset and pull out each 0.5 degree tilt for analysis
        try:
            radar1 = pyart.io.nexrad_archive.read_nexrad_archive(folder+'/'+radar_file)
        except:
            print('bad radar file')
            continue
        #Local file option
        print('File Reading')
        #Make sure the file isn't a strange format
        if radar1.nsweeps > zero_z_trigger:
            continue
            
        for i in range(radar1.nsweeps):
            print('in loop')
            print(radar1.nsweeps)
            try:
                radar4 = radar1.extract_sweeps([i])
            except:
                print('bad file')
            #Checking to make sure the tilt in question has all needed data and is the right elevation
            if ((np.mean(radar4.elevation['data']) < .65) and (np.max(np.asarray(radar4.fields['differential_reflectivity']['data'])) != np.min(np.asarray(radar4.fields['differential_reflectivity']['data'])))):
                n = n+1

                #Calling ungridded_section; Pulling apart radar sweeps and creating ungridded data arrays
                [radar,n,range_2d,ungrid_lons,ungrid_lats] = quality_control_arcalg(radar4,n,calibration)

                time_start = netCDF4.num2date(radar.time['data'][0], radar.time['units'])
                object_number=0.0
                month = time_start.month
                if month < 10:
                    month = '0'+str(month)
                hour = time_start.hour
                if hour < 10:
                    hour = '0'+str(hour)
                minute = time_start.minute
                if minute < 10:
                    minute = '0'+str(minute)
                day = time_start.day
                if day < 10:
                    day = '0'+str(day)
                time_beg = time_start - timedelta(minutes=0.1)
                time_end = time_start + timedelta(minutes=GR_mins)
                sec_beg = time_beg.second
                sec_end = time_end.second
                min_beg = time_beg.minute
                min_end = time_end.minute
                h_beg = time_beg.hour
                h_end = time_end.hour
                d_beg = time_beg.day
                d_end = time_end.day
                if sec_beg < 10:
                    sec_beg = '0'+str(sec_beg)
                if sec_end < 10:
                    sec_end = '0'+str(sec_end)
                if min_beg < 10:
                    min_beg = '0'+str(min_beg)
                if min_end < 10:
                    min_end = '0'+str(min_end)
                if h_beg < 10:
                    h_beg = '0'+str(h_beg)
                if h_end < 10:
                    h_end = '0'+str(h_end)
                if d_beg < 10:
                    d_beg = '0'+str(d_beg)
                if d_end < 10:
                    d_end = '0'+str(d_end)

                #Calling kdp_section; Using NWS method, creating ungridded, smoothed KDP field
                kdp_nwsdict = kdp_genesis(radar)

                #Add field to radar
                radar.add_field('KDP', kdp_nwsdict)
                kdp_ungridded_nws = radar.fields['KDP']['data']


                #Calling grid_section; Now let's grid the data on a ~250 m x 250 m grid
                [REF,KDP,CC,ZDRmasked1,REFmasked,KDPmasked,rlons,rlats,rlons_2d,rlats_2d,cenlat,cenlon] = gridding_arcalg(radar)

                #Calling gradient_section; Determining gradient direction and masking some Zhh and Zdr grid fields
                [grad_mag,grad_ffd,ZDRmasked] = grad_mask_arcalg(REFmasked,REF,storm_relative_dir,ZDRmasked1,CC)

                #Let's create a field for inferred hail
                #Commenting out for the moment
#                 REF_Hail = np.copy(REFmasked)
#                 REF_Hail1 = ma.masked_where(ZDRmasked1 > 1.0, REF_Hail)
#                 REF_Hail2 = ma.masked_where(CC > 1.0, REF_Hail1)
#                 REF_Hail2 = ma.filled(REF_Hail2, fill_value = 1)

                #Let's set up the map projection!
                crs = ccrs.LambertConformal(central_longitude=-100.0, central_latitude=45.0)

                #Set up our array of latitude and longitude values and transform our data to the desired projection.
                tlatlons = crs.transform_points(ccrs.LambertConformal(central_longitude=265, central_latitude=25, standard_parallels=(25.,25.)),rlons[0,:,:],rlats[0,:,:])
                tlons = tlatlons[:,:,0]
                tlats = tlatlons[:,:,1]

                #Limit the extent of the map area, must convert to proper coords.
                LL = (cenlon-1.0,cenlat-1.0,ccrs.PlateCarree())
                UR = (cenlon+1.0,cenlat+1.0,ccrs.PlateCarree())
                print(LL)

                #Get data to plot state and province boundaries
                states_provinces = cfeature.NaturalEarthFeature(
                        category='cultural',
                        name='admin_1_states_provinces_lakes',
                        scale='50m',
                        facecolor='none')
                #Make sure these shapefiles are in the same directory as the script
                #fname = 'cb_2016_us_county_20m/cb_2016_us_county_20m.shp'
                #fname2 = 'cb_2016_us_state_20m/cb_2016_us_state_20m.shp'
                #counties = ShapelyFeature(Reader(fname).geometries(),ccrs.PlateCarree(), facecolor = 'none', edgecolor = 'black')
                #states = ShapelyFeature(Reader(fname2).geometries(),ccrs.PlateCarree(), facecolor = 'none', edgecolor = 'black')

                #Create a figure and plot up the initial data and contours for the algorithm
                fig=plt.figure(n,figsize=(30.,25.))
                ax = plt.subplot(111,projection=ccrs.PlateCarree())
                ax.coastlines('50m',edgecolor='black',linewidth=0.75)
                #ax.add_feature(counties, edgecolor = 'black', linewidth = 0.5)
                #ax.add_feature(states, edgecolor = 'black', linewidth = 1.5)
                ax.set_extent([LL[0],UR[0],LL[1],UR[1]])
                REFlevels = np.arange(20,73,2)

                #Options for Z backgrounds/contours
                #refp = ax.pcolormesh(ungrid_lons, ungrid_lats, ref_c, cmap=plt.cm.gist_ncar, vmin = 10, vmax = 73)
                #refp = ax.pcolormesh(ungrid_lons, ungrid_lats, ref_ungridded_base, cmap='HomeyerRainbow', vmin = 10, vmax = 73)
                #refp = ax.pcolormesh(rlons_2d, rlats_2d, REFrmasked, cmap=pyart.graph.cm_colorblind.HomeyerRainbow, vmin = 10, vmax = 73)
                refp2 = ax.contour(rlons_2d, rlats_2d, REFmasked, [40], colors='grey', linewidths=5, zorder=1)
                #refp3 = ax.contour(rlons_2d, rlats_2d, REFmasked, [45], color='r')
                #plt.contourf(rlons_2d, rlats_2d, ZDR_sum_stuff, depth_levels, cmap=plt.cm.viridis)

                #Option to have a ZDR background instead of Z:
                #zdrp = ax.pcolormesh(ungrid_lons, ungrid_lats, zdr_c, cmap=plt.cm.nipy_spectral, vmin = -2, vmax = 6)

                #Storm tracking algorithm starts here
                #Reflectivity smoothed for storm tracker
                smoothed_ref = ndi.gaussian_filter(REFmasked, sigma = 3, order = 0)
                #1st Z contour plotted
                refc = ax.contour(rlons[0,:,:],rlats[0,:,:],smoothed_ref,REFlev, alpha=.01)

                #Set up projection for area calculations
                proj_old = partial(pyproj.transform, pyproj.Proj(init='epsg:4326'),
                           pyproj.Proj(init='epsg:3857'))
                proj = partial(pyproj.transform, pyproj.Proj(init='epsg:4326'),
                           pyproj.Proj("+proj=aea +lat_1=37.0 +lat_2=41.0 +lat_0=39.0 +lon_0=-106.55"))
                
                #Main part of storm tracking algorithm starts by looping through all contours looking for Z centroids
                #This method for breaking contours into polygons based on this stack overflow tutorial:
                #https://gis.stackexchange.com/questions/99917/converting-matplotlib-contour-objects-to-shapely-objects
                #Calling stormid_section
                [storm_ids,max_lons_c,max_lats_c,ref_areas,storm_index] = storm_objects(refc,proj_old,REFlev,REFlev1,big_storm,smoothed_ref,ax,rlons,rlats,storm_index,tracking_index,scan_index,tracks_dataframe, track_dis)

                #Setup tracking index for storm of interest
                tracking_ind=np.where(np.asarray(storm_ids)==storm_to_track)[0]
                max_lons_c = np.asarray(max_lons_c)
                max_lats_c = np.asarray(max_lats_c)
                ref_areas = np.asarray(ref_areas)
                #Create the ZDR and KDP contours which will later be broken into polygons
                if np.max(ZDRmasked) > zdrlev:
                    zdrc = ax.contour(rlons[0,:,:],rlats[0,:,:],ZDRmasked,zdrlev,linewidths = 2, colors='purple', alpha = .5)
                else:
                    zdrc=[]
                if np.max(KDPmasked) > kdplev:
                    kdpc = ax.contour(rlons[0,:,:],rlats[0,:,:],KDPmasked,kdplev,linewidths = 2, colors='green', alpha = 0.01)
                else:
                    kdpc=[]
#                 if np.max(REF_Hail2) > 50.0:
#                     hailc = ax.contour(rlons[0,:,:],rlats[0,:,:],REF_Hail2,[50],linewidths = 4, colors='pink', alpha = 0.01)
#                 else:
#                     hailc=[]
                if np.max(REFmasked) > 35.0:
                    zhhc = ax.contour(rlons[0,:,:],rlats[0,:,:],REFmasked,[35.0],linewidths = 3,colors='orange', alpha = 0.8)
                else:
                    zhhc=[]
                plt.contour(ungrid_lons, ungrid_lats, range_2d, [73000], linewidths=7, colors='r')
                plt.savefig('testfig.png')
                print('Testfig Saved')

                if len(max_lons_c) > 0:
                    #Calling zdr_arc_section; Create ZDR arc objects using a similar method as employed in making the storm objects
                    [zdr_storm_lon,zdr_storm_lat,zdr_dist,zdr_forw,zdr_back,zdr_areas,zdr_centroid_lon,zdr_centroid_lat,zdr_mean,zdr_cc_mean,zdr_max,zdr_masks,zdr_outlines,ax,f] = zdrarc(zdrc,ZDRmasked,CC,REF,grad_ffd,grad_mag,KDP,forest_loaded,ax,f,time_start,month,d_beg,h_beg,min_beg,sec_beg,d_end,h_end,min_end,sec_end,rlons,rlats,max_lons_c,max_lats_c,zdrlev,proj,storm_relative_dir,Outer_r,Inner_r,tracking_ind)


                    #Calling hail_section; Identify Hail core objects in a similar way to the ZDR arc objects
#                     [hail_areas,hail_centroid_lon,hail_centroid_lat,hail_storm_lon,hail_storm_lat,ax,f] = hail_objects(hailc,REF_Hail2,ax,f,time_start,month,d_beg,h_beg,min_beg,sec_beg,d_end,h_end,min_end,sec_end,rlons,rlats,max_lons_c,max_lats_c,proj)


                    #Calling zhh_section; Identify 35dBz storm area in a similar way to the ZDR arc objects
                    [zhh_areas,zhh_centroid_lon,zhh_centroid_lat,zhh_storm_lon,zhh_storm_lat,zhh_max,zhh_core_avg] = zhh_objects(zhhc,REFmasked,rlons,rlats,max_lons_c,max_lats_c,proj)


                    #Calling kdpfoot_section; Identify KDP foot objects in a similar way to the ZDR arc objects
                    [kdp_areas,kdp_centroid_lon,kdp_centroid_lat,kdp_storm_lon,kdp_storm_lat,kdp_max,ax,f] = kdp_objects(kdpc,KDPmasked,ax,f,time_start,month,d_beg,h_beg,min_beg,sec_beg,d_end,h_end,min_end,sec_end,rlons,rlats,max_lons_c,max_lats_c,kdplev,proj)


                    #Consolidating the arc objects associated with each storm:
                    zdr_areas_arr = np.zeros((len(zdr_areas)))
                    zdr_max_arr = np.zeros((len(zdr_max)))
                    zdr_mean_arr = np.zeros((len(zdr_mean)))                    
                    for i in range(len(zdr_areas)):
                        zdr_areas_arr[i] = zdr_areas[i].magnitude
                        zdr_max_arr[i] = zdr_max[i]
                        zdr_mean_arr[i] = zdr_mean[i]
                    zdr_centroid_lons = np.asarray(zdr_centroid_lon)
                    zdr_centroid_lats = np.asarray(zdr_centroid_lat)
                    zdr_con_areas = []
                    zdr_con_maxes = []
                    zdr_con_means = []
                    zdr_con_centroid_lon = []
                    zdr_con_centroid_lat = []
                    zdr_con_max_lon = []
                    zdr_con_max_lat = []
                    zdr_con_storm_lon = []
                    zdr_con_storm_lat = []
                    zdr_con_masks = []
                    zdr_con_dev = []
                    zdr_con_10max = []
                    zdr_con_mode = []
                    zdr_con_median = []
                    zdr_masks = np.asarray(zdr_masks)

                    #Consolidate KDP objects as well
                    kdp_areas_arr = np.zeros((len(kdp_areas)))
                    kdp_max_arr = np.zeros((len(kdp_max)))
                    for i in range(len(kdp_areas)):
                        kdp_areas_arr[i] = kdp_areas[i].magnitude
                        kdp_max_arr[i] = kdp_max[i]
                    kdp_centroid_lons = np.asarray(kdp_centroid_lon)
                    kdp_centroid_lats = np.asarray(kdp_centroid_lat)
                    kdp_con_areas = []
                    kdp_con_maxes = []
                    kdp_con_centroid_lon = []
                    kdp_con_centroid_lat = []
                    kdp_con_max_lon = []
                    kdp_con_max_lat = []
                    kdp_con_storm_lon = []
                    kdp_con_storm_lat = []

                    #Consolidate Hail objects as well
#                     hail_areas_arr = np.zeros((len(hail_areas)))
#                     for i in range(len(hail_areas)):
#                         hail_areas_arr[i] = hail_areas[i].magnitude
#                     hail_centroid_lons = np.asarray(hail_centroid_lon)
#                     hail_centroid_lats = np.asarray(hail_centroid_lat)
#                     hail_con_areas = []
#                     hail_con_centroid_lon = []
#                     hail_con_centroid_lat = []
#                     hail_con_storm_lon = []
#                     hail_con_storm_lat = []

                    #Consolidate Zhh objects as well
                    zhh_areas_arr = np.zeros((len(zhh_areas)))
                    zhh_max_arr = np.zeros((len(zhh_max)))
                    zhh_core_avg_arr = np.zeros((len(zhh_core_avg)))
                    for i in range(len(zhh_areas)):
                        zhh_areas_arr[i] = zhh_areas[i].magnitude
                        zhh_max_arr[i] = zhh_max[i]
                        zhh_core_avg_arr[i] = zhh_core_avg[i]
                    zhh_centroid_lons = np.asarray(zhh_centroid_lon)
                    zhh_centroid_lats = np.asarray(zhh_centroid_lat)
                    zhh_con_areas = []
                    zhh_con_maxes = []
                    zhh_con_core_avg = []
                    zhh_con_centroid_lon = []
                    zhh_con_centroid_lat = []
                    zhh_con_max_lon = []
                    zhh_con_max_lat = []
                    zhh_con_storm_lon = []
                    zhh_con_storm_lat = []

                    for i in enumerate(max_lons_c):
                        try:
                            #Find the arc objects associated with this storm:
                            zdr_objects_lons = zdr_centroid_lons[np.where(zdr_storm_lon == max_lons_c[i[0]])]
                            zdr_objects_lats = zdr_centroid_lats[np.where(zdr_storm_lon == max_lons_c[i[0]])]

                            #Get the sum of their areas
                            zdr_con_areas.append(np.sum(zdr_areas_arr[np.where(zdr_storm_lon == max_lons_c[i[0]])]))
                            #print("consolidated area", np.sum(zdr_areas_arr[np.where(zdr_storm_lon == max_lons_c[i[0]])]))
                            zdr_con_maxes.append(np.max(zdr_max_arr[np.where(zdr_storm_lon == max_lons_c[i[0]])]))
                            #print("consolidated max", np.max(zdr_areas_arr[np.where(zdr_storm_lon == max_lons_c[i[0]])]))
                            zdr_con_means.append(np.mean(zdr_mean_arr[np.where(zdr_storm_lon == max_lons_c[i[0]])]))
                            #print("consolidated mean", np.mean(zdr_areas_arr[np.where(zdr_storm_lon == max_lons_c[i[0]])]))
                            zdr_con_max_lon.append(rlons_2d[np.where(ZDRmasked==np.max(zdr_max_arr[np.where(zdr_storm_lon == max_lons_c[i[0]])]))])
                            zdr_con_max_lat.append(rlats_2d[np.where(ZDRmasked==np.max(zdr_max_arr[np.where(zdr_storm_lon == max_lons_c[i[0]])]))])

                            #Find the actual centroids
                            weighted_lons = zdr_objects_lons * zdr_areas_arr[np.where(zdr_storm_lon == max_lons_c[i[0]])]
                            zdr_con_centroid_lon.append(np.sum(weighted_lons) / np.sum(zdr_areas_arr[np.where(zdr_storm_lon == max_lons_c[i[0]])]))
                            weighted_lats = zdr_objects_lats * zdr_areas_arr[np.where(zdr_storm_lon == max_lons_c[i[0]])]
                            zdr_con_centroid_lat.append(np.sum(weighted_lats) / np.sum(zdr_areas_arr[np.where(zdr_storm_lon == max_lons_c[i[0]])]))
                            zdr_con_storm_lon.append(max_lons_c[i[0]])
                            zdr_con_storm_lat.append(max_lats_c[i[0]])
                            zdr_con_masks.append(np.sum(zdr_masks[np.where(zdr_storm_lon == max_lons_c[i[0]])],axis=0, dtype=bool))
                            mask_con = np.sum(zdr_masks[np.where(zdr_storm_lon == max_lons_c[i[0]])], axis=0, dtype=bool)
                            zdr_con_dev.append(np.std(ZDRmasked[mask_con]))
                            ZDRsorted = np.sort(ZDRmasked[mask_con])[::-1]
                            zdr_con_10max.append(np.mean(ZDRsorted[0:10]))
                            zdr_con_mode.append(stats.mode(ZDRmasked[mask_con]))
                            zdr_con_median.append(np.median(ZDRmasked[mask_con]))
                        except:
                            zdr_con_maxes.append(0)
                            zdr_con_means.append(0)
                            zdr_con_centroid_lon.append(0)
                            zdr_con_centroid_lat.append(0)
                            zdr_con_max_lon.append(0)
                            zdr_con_max_lat.append(0)
                            zdr_con_storm_lon.append(max_lons_c[i[0]])
                            zdr_con_storm_lat.append(max_lats_c[i[0]])
                            zdr_con_masks.append(0)
                            zdr_con_dev.append(0)
                            zdr_con_10max.append(0)
                            zdr_con_mode.append(0)
                            zdr_con_median.append(0)

                        try:
                            #Find the kdp objects associated with this storm:
                            kdp_objects_lons = kdp_centroid_lons[np.where(kdp_storm_lon == max_lons_c[i[0]])]
                            kdp_objects_lats = kdp_centroid_lats[np.where(kdp_storm_lon == max_lons_c[i[0]])]

                            #Get the sum of their areas
                            kdp_con_areas.append(np.sum(kdp_areas_arr[np.where(kdp_storm_lon == max_lons_c[i[0]])]))
                            kdp_con_maxes.append(np.max(kdp_max_arr[np.where(kdp_storm_lon == max_lons_c[i[0]])]))
                            kdp_con_max_lon.append(rlons_2d[np.where(KDPmasked==np.max(kdp_max_arr[np.where(kdp_storm_lon == max_lons_c[i[0]])]))])
                            kdp_con_max_lat.append(rlats_2d[np.where(KDPmasked==np.max(kdp_max_arr[np.where(kdp_storm_lon == max_lons_c[i[0]])]))])
                            #Find the actual centroids
                            weighted_lons_kdp = kdp_objects_lons * kdp_areas_arr[np.where(kdp_storm_lon == max_lons_c[i[0]])]
                            kdp_con_centroid_lon.append(np.sum(weighted_lons_kdp) / np.sum(kdp_areas_arr[np.where(kdp_storm_lon == max_lons_c[i[0]])]))
                            weighted_lats_kdp = kdp_objects_lats * kdp_areas_arr[np.where(kdp_storm_lon == max_lons_c[i[0]])]
                            kdp_con_centroid_lat.append(np.sum(weighted_lats_kdp) / np.sum(kdp_areas_arr[np.where(kdp_storm_lon == max_lons_c[i[0]])]))
                            kdp_con_storm_lon.append(max_lons_c[i[0]])
                            kdp_con_storm_lat.append(max_lats_c[i[0]])
                        except:
                            kdp_con_maxes.append(0)
                            kdp_con_max_lon.append(0)
                            kdp_con_max_lat.append(0)
                            kdp_con_centroid_lon.append(0)
                            kdp_con_centroid_lat.append(0)
                            kdp_con_storm_lon.append(0)
                            kdp_con_storm_lat.append(0)

#                         try:
#                             #Find the hail core objects associated with this storm:
#                             hail_objects_lons = hail_centroid_lons[np.where(hail_storm_lon == max_lons_c[i[0]])]
#                             hail_objects_lats = hail_centroid_lats[np.where(hail_storm_lon == max_lons_c[i[0]])]
#                             #Get the sum of their areas
#                             hail_con_areas.append(np.sum(hail_areas_arr[np.where(hail_storm_lon == max_lons_c[i[0]])]))
#                             #Find the actual centroids
#                             weighted_lons_hail = hail_objects_lons * hail_areas_arr[np.where(hail_storm_lon == max_lons_c[i[0]])]
#                             hail_con_centroid_lon.append(np.sum(weighted_lons_hail) / np.sum(hail_areas_arr[np.where(hail_storm_lon == max_lons_c[i[0]])]))
#                             weighted_lats_hail = hail_objects_lats * hail_areas_arr[np.where(hail_storm_lon == max_lons_c[i[0]])]
#                             hail_con_centroid_lat.append(np.sum(weighted_lats_hail) / np.sum(hail_areas_arr[np.where(hail_storm_lon == max_lons_c[i[0]])]))
#                             hail_con_storm_lon.append(max_lons_c[i[0]])
#                             hail_con_storm_lat.append(max_lats_c[i[0]])
#                         except:
#                             hail_con_centroid_lon.append(0)
#                             hail_con_centroid_lat.append(0)
#                             hail_con_storm_lon.append(0)
#                             hail_con_storm_lat.append(0)

                        try:
                            #Find the zhh objects associated with this storm:
                            zhh_objects_lons = zhh_centroid_lons[np.where(zhh_storm_lon == max_lons_c[i[0]])]
                            zhh_objects_lats = zhh_centroid_lats[np.where(zhh_storm_lon == max_lons_c[i[0]])]
                            #Get the sum of their areas
                            zhh_con_areas.append(np.sum(zhh_areas_arr[np.where(zhh_storm_lon == max_lons_c[i[0]])]))
                            zhh_con_maxes.append(np.max(zhh_max_arr[np.where(zhh_storm_lon == max_lons_c[i[0]])]))
                            zhh_con_core_avg.append(np.max(zhh_core_avg_arr[np.where(zhh_storm_lon == max_lons_c[i[0]])]))
                            zhh_con_max_lon.append(rlons_2d[np.where(REFmasked==np.max(zhh_max_arr[np.where(zhh_storm_lon == max_lons_c[i[0]])]))])
                            zhh_con_max_lat.append(rlats_2d[np.where(REFmasked==np.max(zhh_max_arr[np.where(zhh_storm_lon == max_lons_c[i[0]])]))])
                            #Find the actual centroids
                            weighted_lons_zhh = zhh_objects_lons * zhh_areas_arr[np.where(zhh_storm_lon == max_lons_c[i[0]])]
                            zhh_con_centroid_lon.append(np.sum(weighted_lons_zhh) / np.sum(zhh_areas_arr[np.where(zhh_storm_lon == max_lons_c[i[0]])]))
                            weighted_lats_zhh = zhh_objects_lats * zhh_areas_arr[np.where(zhh_storm_lon == max_lons_c[i[0]])]
                            zhh_con_centroid_lat.append(np.sum(weighted_lats_zhh) / np.sum(zhh_areas_arr[np.where(zhh_storm_lon == max_lons_c[i[0]])]))
                            zhh_con_storm_lon.append(max_lons_c[i[0]])
                            zhh_con_storm_lat.append(max_lats_c[i[0]])
                        except:
                            zhh_con_maxes.append(0)
                            zhh_con_core_avg.append(0)
                            zhh_con_max_lon.append(0)
                            zhh_con_max_lat.append(0)
                            zhh_con_centroid_lon.append(0)
                            zhh_con_centroid_lat.append(0)
                            zhh_con_storm_lon.append(0)
                            zhh_con_storm_lat.append(0)

                        #Calculate KDP-ZDR separation
        #             kdp_con_centroid_lons1 = np.asarray(kdp_con_centroid_lon)
        #             kdp_con_centroid_lats1 = np.asarray(kdp_con_centroid_lat)
        #             zdr_con_centroid_lons1 = np.asarray(zdr_con_centroid_lon)
        #             zdr_con_centroid_lats1 = np.asarray(zdr_con_centroid_lat)
        #             #Eliminate consolidated arcs smaller than a specified area
        #             area = 2 #km*2
        #             zdr_con_areas_arr = np.asarray(zdr_con_areas)
        #             zdr_con_centroid_lats = zdr_con_centroid_lats1[zdr_con_areas_arr > area]
        #             zdr_con_centroid_lons = zdr_con_centroid_lons1[zdr_con_areas_arr > area]
        #             kdp_con_centroid_lats = kdp_con_centroid_lats1[zdr_con_areas_arr > area]
        #             kdp_con_centroid_lons = kdp_con_centroid_lons1[zdr_con_areas_arr > area]
        #             zdr_con_max_lons1 = np.asarray(zdr_con_max_lon)[zdr_con_areas_arr > area]
        #             zdr_con_max_lats1 = np.asarray(zdr_con_max_lat)[zdr_con_areas_arr > area]
        #             kdp_con_max_lons1 = np.asarray(kdp_con_max_lon)[zdr_con_areas_arr > area]
        #             kdp_con_max_lats1 = np.asarray(kdp_con_max_lat)[zdr_con_areas_arr > area]
        #             zdr_con_max1 = np.asarray(zdr_con_maxes)[zdr_con_areas_arr > area]
        #             zdr_con_areas1 = zdr_con_areas_arr[zdr_con_areas_arr > area]
                    kdp_con_centroid_lat = np.asarray(kdp_con_centroid_lat)
                    kdp_con_centroid_lon = np.asarray(kdp_con_centroid_lon)
                    zdr_con_centroid_lat = np.asarray(zdr_con_centroid_lat)
                    zdr_con_centroid_lon = np.asarray(zdr_con_centroid_lon)

                    kdp_inds = np.where(kdp_con_centroid_lat*zdr_con_centroid_lat > 0)
                    distance_kdp_zdr = g.inv(kdp_con_centroid_lon[kdp_inds], kdp_con_centroid_lat[kdp_inds], zdr_con_centroid_lon[kdp_inds], zdr_con_centroid_lat[kdp_inds])
                    dist_kdp_zdr = distance_kdp_zdr[2] / 1000.
                    #Now make an array for the distances which will have the same shape as the lats to prevent errors
                    shaped_dist = np.zeros((np.shape(zdr_con_areas)))
                    shaped_dist[kdp_inds] = dist_kdp_zdr

                    #Get separation angle for KDP-ZDR centroids
                    back_k = distance_kdp_zdr[1]
                    for i in range(back_k.shape[0]):
                        if distance_kdp_zdr[1][i] < 0:
                            back_k[i] = distance_kdp_zdr[1][i] + 360

                    forw_k = np.abs(back_k - storm_relative_dir)
                    rawangle_k = back_k - storm_relative_dir
                    #Account for weird angles
                    for i in range(back_k.shape[0]):
                        if forw_k[i] > 180:
                            forw_k[i] = 360 - forw_k[i]
                            rawangle_k[i] = (360-forw_k[i])*(-1)

                    rawangle_k = rawangle_k*(-1)

                    #Now make an array for the distances which will have the same shape as the lats to prevent errors
                    shaped_ang = np.zeros((np.shape(zdr_con_areas)))
                    shaped_ang[kdp_inds] = rawangle_k
                    shaped_ang = (180-np.abs(shaped_ang))*(shaped_ang/np.abs(shaped_ang))

                    new_angle_all = shaped_ang + storm_relative_dir

                    shaped_ang = (new_angle_all - Bunkers_m)* (-1)

                    shaped_ang = 180 - shaped_ang

                    ###Now let's consolidate everything to fit the Pandas dataframe!
                    p_zdr_areas = []
                    p_zdr_maxes = []
                    p_zdr_means = []
                    p_zdr_devs = []
                    p_zdr_10max = []
                    p_zdr_mode = []
                    p_zdr_median = []
                   # p_hail_areas = []
                    p_zhh_areas = []
                    p_zhh_maxes = []
                    p_zhh_core_avgs = []
                    p_separations = []
                    p_sp_angle = []
                    for storm in enumerate(max_lons_c):
                        matching_ind = np.flatnonzero(np.isclose(max_lons_c[storm[0]], zdr_con_storm_lon, rtol=1e-05))
                        if matching_ind.shape[0] > 0:
                            p_zdr_areas.append((zdr_con_areas[matching_ind[0]]))
                            p_zdr_maxes.append((zdr_con_maxes[matching_ind[0]]))
                            p_zdr_means.append((zdr_con_means[matching_ind[0]]))
                            p_zdr_devs.append((zdr_con_dev[matching_ind[0]]))
                            p_zdr_10max.append((zdr_con_10max[matching_ind[0]]))
                            p_zdr_mode.append((zdr_con_mode[matching_ind[0]]))
                            p_zdr_median.append((zdr_con_median[matching_ind[0]]))
                            p_separations.append((shaped_dist[matching_ind[0]]))
                            p_sp_angle.append((shaped_ang[matching_ind[0]]))
                        else:
                            p_zdr_areas.append((0))
                            p_zdr_maxes.append((0))
                            p_zdr_means.append((0))
                            p_zdr_devs.append((0))
                            p_zdr_10max.append((0))
                            p_zdr_mode.append((0))
                            p_zdr_median.append((0))
                            p_separations.append((0))
                            p_sp_angle.append((0))

#                         matching_ind_hail = np.flatnonzero(np.isclose(max_lons_c[storm[0]], hail_con_storm_lon, rtol=1e-05))
#                         if matching_ind_hail.shape[0] > 0:
#                             p_hail_areas.append((hail_con_areas[matching_ind_hail[0]]))
#                         else:
#                             p_hail_areas.append((0))

                        matching_ind_zhh = np.flatnonzero(np.isclose(max_lons_c[storm[0]],zhh_con_storm_lon, rtol=1e-05))
                        if matching_ind_zhh.shape[0] > 0:
                            p_zhh_maxes.append((zhh_con_maxes[matching_ind_zhh[0]]))
                            p_zhh_areas.append((zhh_con_areas[matching_ind_zhh[0]]))
                            p_zhh_core_avgs.append((zhh_con_core_avg[matching_ind_zhh[0]]))
                        else:
                            p_zhh_areas.append((0))
                            p_zhh_maxes.append((0))
                            p_zhh_core_avgs.append((0))

                    #Now start plotting stuff!
                    if np.asarray(zdr_centroid_lon).shape[0] > 0:
                        ax.scatter(zdr_centroid_lon, zdr_centroid_lat, marker = '*', s = 100, color = 'black', zorder = 10, transform=ccrs.PlateCarree())
                    if np.asarray(kdp_centroid_lon).shape[0] > 0:
                        ax.scatter(kdp_centroid_lon, kdp_centroid_lat, marker = '^', s = 100, color = 'black', zorder = 10, transform=ccrs.PlateCarree())
                    #Uncomment to print all object areas
                    #for i in enumerate(zdr_areas):
                    #    plt.text(zdr_centroid_lon[i[0]]+.016, zdr_centroid_lat[i[0]]+.016, "%.2f km^2" %(zdr_areas[i[0]].magnitude), size = 23)
                        #plt.text(zdr_centroid_lon[i[0]]+.016, zdr_centroid_lat[i[0]]+.016, "%.2f km^2 / %.2f km / %.2f dB" %(zdr_areas[i[0]].magnitude, zdr_dist[i[0]], zdr_forw[i[0]]), size = 23)
                        #plt.annotate(zdr_areas[i[0]], (zdr_centroid_lon[i[0]],zdr_centroid_lat[i[0]]))
                    #ax.contourf(rlons[0,:,:],rlats[0,:,:],KDPmasked,KDPlevels1,linewide = .01, colors ='b', alpha = .5)
                    #plt.tight_layout()
                    #plt.savefig('ZDRarcannotated.png')
                    storm_times = []
                    for l in range(len(max_lons_c)):
                        storm_times.append((time_start))
                    tracking_index = tracking_index + 1

                #If there are no storms, set everything to empty arrays!
                else:
                    storm_ids = []
                    storm_ids = []
                    max_lons_c = []
                    max_lats_c = []
                    p_zdr_areas = []
                    p_zdr_maxes = []
                    p_zdr_means = []
                    p_zdr_devs = []
                    p_zdr_10max = []
                    p_zdr_mode = []
                    p_zdr_median = []
                    #p_hail_areas = []
                    p_zhh_areas = []
                    p_zhh_maxes = []
                    p_zhh_core_avgs = []
                    p_separations = []
                    p_sp_angle = []
                    zdr_con_areas1 = []
                    storm_times = time_start
                #Now record all data in a Pandas dataframe.
                new_cells = pd.DataFrame({
                    'scan': scan_index,
                    'storm_id' : storm_ids,
                    'storm_id1' : storm_ids,
                    'storm_lon' : max_lons_c,
                    'storm_lat' : max_lats_c,
                    'zdr_area' : p_zdr_areas,
                    'zdr_max' : p_zdr_maxes,
                    'zdr_mean' : p_zdr_means,
                    'zdr_std' : p_zdr_devs,
                    'zdr_10max' : p_zdr_10max,
                    'zdr_mode' : p_zdr_mode,
                    'zdr_median' : p_zdr_median,
                    #'hail_area' : p_hail_areas,
                    'zhh_area' : p_zhh_areas,
                    'zhh_max' : p_zhh_maxes,
                    'zhh_core_avg' : p_zhh_core_avgs,
                    'kdp_zdr_sep' : p_separations,
                    'kdp_zdr_angle' : p_sp_angle,
                    'times' : storm_times
                })
                new_cells.set_index(['scan', 'storm_id'], inplace=True)
                if scan_index == 0:
                    tracks_dataframe = new_cells
                else:
                    tracks_dataframe = tracks_dataframe.append(new_cells)
                n = n+1
                scan_index = scan_index + 1

                #Plot the consolidated stuff!
                #Write some text objects for the ZDR arc attributes to add to the placefile
                f.write('TimeRange: '+str(time_start.year)+'-'+str(month)+'-'+str(d_beg)+'T'+str(h_beg)+':'+str(min_beg)+':'+str(sec_beg)+'Z '+str(time_start.year)+'-'+str(month)+'-'+str(d_end)+'T'+str(h_end)+':'+str(min_end)+':'+str(sec_end)+'Z')
                f.write('\n')
                f.write("Color: 139 000 000 \n")
                f.write('Font: 1, 30, 1,"Arial" \n')
                for y in range(len(p_zdr_areas)):
                    #f.write('Text: '+str(max_lats_c[y])+','+str(max_lons_c[y])+', 1, "X"," Arc Area: '+str(p_zdr_areas[y])+'\\n Arc Mean: '+str(p_zdr_means[y])+'\\n KDP-ZDR Separation: '+str(p_separations[y])+'\\n Separation Angle: '+str(p_sp_angle[y])+'" \n')
                    f.write('Text: '+str(max_lats_c[y])+','+str(max_lons_c[y])+', 1, "X"," Arc Area: %.2f km^2 \\n Arc Mean: %.2f dB \\n Arc 10 Max Mean: %.2f dB \\n KDP-ZDR Separation: %.2f km \\n Separation Angle: %.2f degrees \n' %(p_zdr_areas[y], p_zdr_means[y], p_zdr_10max[y], p_separations[y], p_sp_angle[y]))



                title_plot = plt.title(station+' Radar Reflectivity, ZDR, and KDP '+str(time_start.year)+'-'+str(time_start.month)+'-'+str(time_start.day)+
                                           ' '+str(hour)+':'+str(minute)+' UTC', size = 25)

                try:
                    plt.plot([zdr_con_centroid_lon[kdp_inds], kdp_con_centroid_lon[kdp_inds]], [zdr_con_centroid_lat[kdp_inds],kdp_con_centroid_lat[kdp_inds]], color = 'k', linewidth = 5, transform=ccrs.PlateCarree())
                except:
                    print('Separation Angle Failure')

                ref_centroid_lon = max_lons_c
                ref_centroid_lat = max_lats_c
                if len(max_lons_c) > 0:
                    ax.scatter(max_lons_c,max_lats_c, marker = "o", color = 'k', s = 500, alpha = .6)
                    for i in enumerate(ref_centroid_lon): 
                        plt.text(ref_centroid_lon[i[0]]+.016, ref_centroid_lat[i[0]]+.016, "storm_id: %.1f" %(storm_ids[i[0]]), size = 25)
                #Comment out this line if not plotting tornado tracks
                #plt.plot([start_torlons, end_torlons], [start_torlats, end_torlats], color = 'purple', linewidth = 5, transform=ccrs.PlateCarree())
                #Add legend stuff
                zdr_outline = mlines.Line2D([], [], color='blue', linewidth = 5, linestyle = 'solid', label='ZDR Arc Outline(Area/Max)')
                kdp_outline = mlines.Line2D([], [], color='green', linewidth = 5,linestyle = 'solid', label='"KDP Foot" Outline')
                separation_vector = mlines.Line2D([], [], color='black', linewidth = 5,linestyle = 'solid', label='KDP/ZDR Centroid Separation Vector (Red Text=Distance)')
                #tor_track = mlines.Line2D([], [], color='purple', linewidth = 5,linestyle = 'solid', label='Tornado Tracks')
                elevation = mlines.Line2D([], [], color='grey', linewidth = 5,linestyle = 'solid', label='Height AGL (m)')

                plt.legend(handles=[zdr_outline, kdp_outline, separation_vector, elevation], loc = 3, fontsize = 25)
                alt_levs = [1000, 2000]
                plt.savefig('Machine_Learning/ARCALG_newforest'+station+str(time_start.year)+str(time_start.month)+str(day)+str(hour)+str(minute)+'.png')
                print('Figure Saved')
                plt.close()
                zdr_out_list.append(zdr_outlines)
                #except:
                #    traceback.print_exc()
                #    continue
    f.close()
    plt.show()
    print('Fin')
    #export_csv = tracks_dataframe.to_csv(r'C:\Users\Nick\Downloads\tracksdataframe.csv',index=None,header=True)
    return tracks_dataframe, zdr_out_list