import numpy as np
import numpy.ma as ma
from scipy import ndimage as ndi

#Now add KDP to the dataset
#Calculate KDP manually following NWS methodology
#First, get the phidp gradient
def kdp_genesis(radar):
      #Inputs,
      #radar: Quality-controlled volume data
      print('KDP Section')
      #Using NWS method, creating ungridded, smoothed KDP field
      #Pulling in required radar fields; Differential phase to have a gradient applied, Correlation coefficient and reflectivity for smoothing
      phidp_ungridded = radar.fields['differential_phase']['data']
      cc_ungridded = radar.fields['cross_correlation_ratio']['data']
      ref_ungridded = radar.fields['reflectivity']['data']
      
      #Creating KDP as the range deriative/gradient of Phidp
      phidp_ungridded = np.asarray(np.gradient(phidp_ungridded))/0.50
      kdp_raw = phidp_ungridded[1,:,:]
      kdp_raw = ma.masked_where(kdp_raw > 40., kdp_raw)

      #Perform NWS outlined smoothing process
      kdp_9s = ndi.uniform_filter1d(kdp_raw, 8, 1)
      kdp_25s = ndi.uniform_filter1d(kdp_raw, 24, 1)
      kdp_9s = ma.masked_where(ref_ungridded < 20, kdp_9s)
      kdp_9s = ma.masked_where(cc_ungridded < 0.90, kdp_9s)
      kdp_25s = ma.masked_where(ref_ungridded < 20, kdp_25s)
      kdp_25s = ma.masked_where(cc_ungridded < 0.90, kdp_25s)
      kdp_9s[ref_ungridded < 40] = kdp_25s[ref_ungridded < 40]
      kdp_9s = ma.masked_where(ref_ungridded < 35, kdp_9s)
      kdp_9s = ma.masked_where(cc_ungridded < 0.90, kdp_9s)
      kdp_8 = np.copy(kdp_9s)
      kdp_8 = ma.masked_where(kdp_8 < 8, kdp_8)
      kdp_8 = ma.masked_where(ref_ungridded > 50, kdp_8)
      kdp_8 = ma.filled(kdp_8, fill_value=-2)
      kdp_9s = ma.masked_where(kdp_8 > 1, kdp_9s)
      kdp_9s = ma.masked_where(ref_ungridded < 20., kdp_9s)

      #Create dictionary
      kdp_nwsdict = {}
      kdp_nwsdict['units'] = 'degrees/km'
      kdp_nwsdict['standard_name'] = 'specific_differential_phase_hv'
      kdp_nwsdict['long_name'] = 'Specific Differential Phase (KDP)'
      kdp_nwsdict['coordinates'] = 'elevation azimuth range'
      kdp_nwsdict['data'] = kdp_9s
      kdp_nwsdict['valid_min'] = 0.0
      kdp_nwsdict['Clipf'] = 3906250000.0

      #Returning variables,
      #kdp_nwsdict: Fully smoothed KDP dictionary to be added to radar data object
      return kdp_nwsdict