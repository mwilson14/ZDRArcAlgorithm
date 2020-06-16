import pyart
import numpy as np
import numpy.ma as ma

def gridding_arcalg(radar):
	#Inputs variables,
	#radar: Quality-controlled volume data
	#Z0C: Height of freezing level in meters
    print('Grid Section')
    #Create grid of data on a ~500m x 500m x 500m grid (500x500x41 Volume array)
    grid = pyart.map.grid_from_radars(
    	(radar,),
    	grid_shape=(41, 500, 500),
    	grid_limits=((0, 10000), (-123000.0, 123000.0), (-123000.0, 123000.0)),
    	fields=['differential_reflectivity','reflectivity','KDP','cross_correlation_ratio'],
    	weighting_function='Barnes')
    
    ZDR = grid.fields['differential_reflectivity']['data'][0,:,:]
    REF = grid.fields['reflectivity']['data'][0,:,:]
    KDP = grid.fields['KDP']['data'][0,:,:]
    CC = grid.fields['cross_correlation_ratio']['data'][0,:,:]
  
    #Masking grid points below 20dBz
    #Creating masked Zdr array for 1km level
    ZDRmasked1 = ma.masked_where(REF < 20, ZDR)
    REFmasked = ma.masked_where(REF < 20, REF)

    #Use a 35 dBZ mask for KDP to only get areas in the storm core. This threshold should be considered more closely
    KDPmasked = ma.masked_where(REF < 35, KDP)
    KDPmasked = ma.filled(KDPmasked, fill_value = -2)

    #Create 2D coordinate arrays used for tracking/place files
    rlons = grid.point_longitude['data']
    rlats = grid.point_latitude['data']
    rlons_2d = rlons[0,:,:]
    rlats_2d = rlats[0,:,:]
    cenlat = radar.latitude['data'][0]
    cenlon = radar.longitude['data'][0]

    #Returning variables,
    #Zint: 1km AFL grid level
    #REF: 1km Reflectivity grid
    #KDP: 1km Specific Differential Phase (KDP) grid
    #CC: 1km Correlation Coefficient (CC) grid
    #CC_c: Zint level CC grid
    #CCall: Full volume CC gridded
    #ZDRmasked1: 1km Differential Reflectiity (Zdr) grid, masked below 20 dBz reflectivity
    #ZDRrmasked1: Full volume Zdr gridded, masked below 20 dBz reflectivity
    #REFmasked: REF masked below 20 dBz
    #REFrmasked: Zint level Reflectivity grid, masked below 20dBz
    #KDPmasked: 1km KDP grid, masked below 50dBz reflectivity and filled with a value of -2.0
    #KDPrmasked: Zint level KDP grid
    #rlons,rlats: Full volume geographic coordinates, longitude and latitude respectively
    #rlons_2d,rlats_2d: Single layer slice of rlons,rlats
    #cenlat,cenlon: Radar latitude and longitude
    return REF,KDP,CC,ZDRmasked1,REFmasked,KDPmasked,rlons,rlats,rlons_2d,rlats_2d,cenlat,cenlon