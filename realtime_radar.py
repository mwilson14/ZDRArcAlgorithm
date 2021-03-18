import urllib.request
import numpy as np
import pyart
import threading
from datetime import datetime, timedelta
import netCDF4

def get_latest_scan(site):    
    print('Beginning file download with urllib2...')

    url = 'https://nomads.ncep.noaa.gov/pub/data/nccf/radar/nexrad_level2/'+str(site)+'/dir.list '
    urllib.request.urlretrieve(url, 'dir_list_radar.txt')

    dir_list_radar = np.genfromtxt('dir_list_radar.txt', delimiter=' ', dtype='str')
    
    print(dir_list_radar[-1, 1])

    print('Beginning radar file download with urllib2...')

    url = 'https://nomads.ncep.noaa.gov/pub/data/nccf/radar/nexrad_level2/'+str(site)+'/' + dir_list_radar[-1, 1]
    urllib.request.urlretrieve(url, 'test_radar_download.bz2')

    radar1 = pyart.io.nexrad_archive.read_nexrad_archive('test_radar_download.bz2')

    zdr_tilts = []
    tilt_vals = []
    #Updating this to account for recently-added sub-0.5 degree tilts
    for i in range(radar1.nsweeps):
        radar2 = radar1.extract_sweeps([i])
        print(np.mean(radar2.elevation['data']))
        tilt_vals.append(np.mean(radar2.elevation['data']))
    tilt_vals = np.asarray(tilt_vals)
    max_tilt = 0.65
    if np.min(tilt_vals) < 0.40:
        max_tilt = np.min(tilt_vals) + 0.07
    print('Max tilt is ', max_tilt)
    for i in range(radar1.nsweeps):
        radar2 = radar1.extract_sweeps([i])
        print(np.mean(radar2.elevation['data']))
        #time_start = netCDF4.num2date(radar2.time['data'][0], radar2.time['units'])
        #print(time_start)
        if ((np.mean(radar2.elevation['data']) < max_tilt) 
            and (np.max(np.asarray(radar2.fields['differential_reflectivity']['data'])) 
                 != np.min(np.asarray(radar2.fields['differential_reflectivity']['data'])))):
            print('howdy')
            zdr_tilts.append(i)
    radar4 = radar1.extract_sweeps([zdr_tilts[-1]])
    time_start = netCDF4.num2date(radar4.time['data'][0], radar4.time['units'])

    return radar4, time_start