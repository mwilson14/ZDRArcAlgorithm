import numpy as np
import numpy.ma as ma
from scipy import ndimage as ndi
from metpy.units import check_units, concatenate, units
from metpy.calc import wind_direction, wind_speed, wind_components

def grad_mask_arcalg(REFmasked,REF,storm_relative_dir,ZDRmasked1,CC):
      #Inputs,
      #Zint: 1km AFL grid level
      #REFmasked: REF masked below 20 dBz
      #REF: 1km Reflectivity grid
      #storm_relative_dir: Vector direction along the reflectivity gradient in the forward flank
      #ZDRmasked1: 1km Differential Reflectiity (Zdr) grid, masked below 20 dBz reflectivity
      #ZDRrmasked1: Full volume Zdr gridded, masked below 20 dBz reflectivity
      #CC: 1km Correlation Coefficient (CC) grid
      #CCall: Full volume CC gridded
      print('Gradient Analysis and Masking')
      #Determining gradient direction and masking some Zhh and Zdr grid fields

      smoothed_ref1 = ndi.gaussian_filter(REFmasked, sigma = 2, order = 0)
      REFgradient = np.asarray(np.gradient(smoothed_ref1))
      REFgradient[0,:,:] = ma.masked_where(REF < 20, REFgradient[0,:,:])
      REFgradient[1,:,:] = ma.masked_where(REF < 20, REFgradient[1,:,:])
      grad_dir1 = wind_direction(REFgradient[1,:,:] * units('m/s'), REFgradient[0,:,:] * units('m/s'))
      grad_mag = wind_speed(REFgradient[1,:,:] * units('m/s'), REFgradient[0,:,:] * units('m/s'))
      grad_dir = ma.masked_where(REF < 20, grad_dir1)

      #Get difference between the gradient direction and the FFD gradient direction calculated earlier
      srdir = storm_relative_dir
      srirad = np.copy(srdir)*units('degrees').to('radian')
      grad_dir = grad_dir*units('degrees').to('radian')
      grad_ffd = np.abs(np.arctan2(np.sin(grad_dir-srirad), np.cos(grad_dir-srirad)))
      grad_ffd = np.asarray(grad_ffd)*units('radian')
      grad_ex = np.copy(grad_ffd)
      grad_ffd = grad_ffd.to('degrees')

      #Mask out areas where the difference between the two is too large and the ZDR is likely not in the forward flank
      ZDRmasked2 = ma.masked_where(grad_ffd > 120 * units('degrees'), ZDRmasked1)
      ZDRmasked = ma.masked_where(CC < .60, ZDRmasked2)

      #Add a fill value for the ZDR mask so that contours will be closed
      ZDRmasked = ma.filled(ZDRmasked, fill_value = -2)

      #Returning variables,
      #grad_mag: Array of wind velocity magnitude along reflectivity gradient
      #grad_ffd: Angle (degrees) used to indicate angular region of supercell containing the forward flank
      #ZDRmasked: Masked array ZDRmasked1 in regions outside the forward flank (grad_ffd) and below 0.6 CC
      #ZDRallmasked: Masked volume array (ZDRrmasked1) below 0.7 CC and filled with -2.0 values
      #ZDRrmasked: ZDRallmasked slice at 1km above freezing level
      return grad_mag,grad_ffd,ZDRmasked