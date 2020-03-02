import numpy as np
from shapely import geometry
from shapely.ops import transform
from metpy.units import atleast_1d, check_units, concatenate, units
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from pyproj import Geod
from metpy.calc import wind_direction, wind_speed, wind_components

def zdrarc(zdrc,ZDRmasked,CC,REF,grad_ffd,grad_mag,KDP,forest_loaded,ax,f,time_start,month,d_beg,h_beg,min_beg,sec_beg,d_end,h_end,min_end,sec_end,rlons,rlats,max_lons_c,max_lats_c,zdrlev,proj,storm_relative_dir,Outer_r,Inner_r,tracking_ind):
    #Inputs,
    #zdrc: Contour of differential reflectivity at zdrlev (typically 1.5 dB)
    #ZDRmasked: Masked array ZDRmasked1 in regions outside the forward flank (grad_ffd) and below 0.6 CC
    #CC: 1km Correlation Coefficient (CC) grid
    #REF: 1km Reflectivity grid
    #grad_ffd: Angle (degrees) used to indicate angular region of supercell containing the forward flank
    #grad_mag: Array of wind velocity magnitude along reflectivity gradient
    #KDP: 1km Specific Differential Phase (KDP) grid
    #forest_loaded: Random forest pickle file for Zdr arcs
    #ax: Subplot object to be built on with each contour
    #f: Placefile, edited throughout the program
    #time_start: Radar file date and time of scan
    #month: Month of case, supplied by user
    #d_beg,h_beg,min_beg,sec_beg,d_end,h_end,min_end,sec_end: Day, hour, minute, second of the beginning and end of a scan
    #rlons,rlats: Full volume geographic coordinates, longitude and latitude respectively
    #max_lons_c,max_lats_c: Centroid coordinates of storm objects
    #zdrlev: User defined value for Zdr contour of arcs
    #proj: Projection of Earth's surface to be used for accurate area and distance calculations
    #storm_relative_dir: Vector direction along the reflectivity gradient in the forward flank
    #Outer_r,Inner_r: Outer and Inner limit of radar range effective for analysis
    #tracking_ind: Index for storm of interest
    zdr_areas = []
    zdr_centroid_lon = []
    zdr_centroid_lat = []
    zdr_mean = []
    zdr_cc_mean = []
    zdr_max = []
    zdr_storm_lon = []
    zdr_storm_lat = []
    zdr_dist = []
    zdr_forw = []
    zdr_back = []
    zdr_masks = []
    zdr_outlines = []
    if np.max(ZDRmasked) > zdrlev:
        #Break contours into polygons using the same method as for reflectivity
        for level in zdrc.collections:
            for contour_poly in level.get_paths(): 
                for n_contour,contour in enumerate(contour_poly.to_polygons()):
                    contour_a = np.asarray(contour[:])
                    xa = contour_a[:,0]
                    ya = contour_a[:,1]
                    polygon_new = geometry.Polygon([(i[0], i[1]) for i in zip(xa,ya)])
                    if n_contour == 0:
                        polygon = polygon_new
                    else:
                        polygon = polygon.difference(polygon_new)

                pr_area = (transform(proj, polygon).area * units('m^2')).to('km^2')
                boundary = np.asarray(polygon.boundary.xy)
                polypath = Path(boundary.transpose())
                coord_map = np.vstack((rlons[0,:,:].flatten(), rlats[0,:,:].flatten())).T 
                mask = polypath.contains_points(coord_map).reshape(rlons[0,:,:].shape)
                mean = np.mean(ZDRmasked[mask])
                mean_cc = np.mean(CC[mask])
                mean_Z = np.mean(REF[mask])
                mean_graddir = np.mean(grad_ffd[mask])
                mean_grad = np.mean(grad_mag[mask])
                mean_kdp = np.mean(KDP[mask])
                #Only select out objects larger than 1 km^2 with high enough CC
                if pr_area > 1 * units('km^2') and mean > zdrlev[0] and mean_cc > .88:
                    g = Geod(ellps='sphere')
                    dist = np.zeros((np.asarray(max_lons_c).shape[0]))
                    forw = np.zeros((np.asarray(max_lons_c).shape[0]))
                    rawangle = np.zeros((np.asarray(max_lons_c).shape[0]))
                    back = np.zeros((np.asarray(max_lons_c).shape[0]))
                    zdr_polypath = polypath
                    #Assign ZDR arc objects to the nearest acceptable storm object
                    for i in range(dist.shape[0]):
                                distance_1 = g.inv(polygon.centroid.x, polygon.centroid.y,
                                                       max_lons_c[i], max_lats_c[i])
                                back[i] = distance_1[1]
                                if distance_1[1] < 0:
                                    back[i] = distance_1[1] + 360
                                forw[i] = np.abs(back[i] - storm_relative_dir)
                                rawangle[i] = back[i] - storm_relative_dir
                                #Account for weird angles
                                if forw[i] > 180:
                                    forw[i] = 360 - forw[i]
                                    rawangle[i] = (360-forw[i])*(-1)
                                dist[i] = distance_1[2]/1000.
                                rawangle[i] = rawangle[i]*(-1)

                    #Pick out only ZDR arc objects with a reasonable probability of actually being in the FFD region
                    #using their location relative to the storm centroid
                    if (forw[np.where(dist == np.min(dist))[0][0]] < 180 and np.min(dist) < Outer_r) or (forw[np.where(dist == np.min(dist))[0][0]] < 140 and np.min(dist) < Inner_r):
                        #Use ML algorithm to eliminate non-arc objects
                        #Get x and y components
                        if (rawangle[np.where(dist == np.min(dist))[0][0]] > 0):
                            directions_raw = 360 - rawangle[np.where(dist == np.min(dist))[0][0]]
                        else:
                            directions_raw = (-1) * rawangle[np.where(dist == np.min(dist))[0][0]]

                        xc, yc = wind_components(np.min(dist)*units('m/s'), directions_raw * units('degree'))
                        ARC_X = np.zeros((1, 12))
                        ARC_X[:,0] = pr_area.magnitude
                        ARC_X[:,1] = np.min(dist)
                        ARC_X[:,2] = np.max(ZDRmasked[mask]) / mean
                        ARC_X[:,3] = (np.max(ZDRmasked[mask]) / mean) * pr_area.magnitude
                        ARC_X[:,4] = mean_cc
                        ARC_X[:,5] = mean_kdp
                        ARC_X[:,6] = mean_Z
                        ARC_X[:,7] = mean_graddir
                        ARC_X[:,8] = mean_grad
                        ARC_X[:,9] = rawangle[np.where(dist == np.min(dist))[0][0]]
                        ARC_X[:,10] = xc
                        ARC_X[:,11] = yc
                        pred_zdr = forest_loaded.predict(ARC_X)
                        if (pred_zdr[0]==1):
                            zdr_storm_lon.append((max_lons_c[np.where(dist == np.min(dist))[0][0]]))
                            zdr_storm_lat.append((max_lats_c[np.where(dist == np.min(dist))[0][0]]))
                            zdr_dist.append(np.min(dist))
                            zdr_forw.append(forw[np.where(dist == np.min(dist))[0][0]])
                            zdr_back.append(back[np.where(dist == np.min(dist))[0][0]])
                            zdr_areas.append((pr_area))
                            zdr_centroid_lon.append((polygon.centroid.x))
                            zdr_centroid_lat.append((polygon.centroid.y))
                            zdr_mean.append((mean))
                            zdr_cc_mean.append((mean_cc))
                            zdr_max.append((np.max(ZDRmasked[mask])))
                            zdr_masks.append(mask)
                            patch = PathPatch(polypath, facecolor='blue', alpha=.6, edgecolor = 'blue', linewidth = 3)
                            ax.add_patch(patch)
                            #Add polygon to placefile
                            f.write('TimeRange: '+str(time_start.year)+'-'+str(month)+'-'+str(d_beg)+'T'+str(h_beg)+':'+str(min_beg)+':'+str(sec_beg)+'Z '+str(time_start.year)+'-'+str(month)+'-'+str(d_end)+'T'+str(h_end)+':'+str(min_end)+':'+str(sec_end)+'Z')
                            f.write('\n')
                            f.write("Color: 000 000 139 \n")
                            f.write('Line: 3, 0, "ZDR Arc Outline" \n')
                            for i in range(len(zdr_polypath.vertices)):
                                f.write("%.5f" %(zdr_polypath.vertices[i][1]))
                                f.write(", ")
                                f.write("%.5f" %(zdr_polypath.vertices[i][0]))
                                f.write('\n')

                            f.write("End: \n \n")
                            if (((max_lons_c[np.where(dist == np.min(dist))[0][0]]) in max_lons_c[tracking_ind]) and ((max_lats_c[np.where(dist == np.min(dist))[0][0]]) in max_lats_c[tracking_ind])):
                                zdr_outlines.append(polypath)

    #Returning Variables,
    #zdr_storm_lon,zdr_storm_lat: Storm object centroids associated with Zdr arcs
    #zdr_dist: Zdr arc centroid distance from associated storm object centroid
    #zdr_forw,zdr_back: Forward and rear angles about the zdr arc region
    #zdr_areas: Zdr arc area
    #zdr_centroid_lon,zdr_centroid_lat: Zdr arc centroid coordinates
    #zdr_mean: Mean value of Zdr within arc
    #zdr_cc_mean: Mean value of CC within arc
    #zdr_max: Maximum value of Zdr within arc
    #zdr_masks: Mask array of zdr arc
    #zdr_outlines: Zdr arc contour array
    #ax: Subplot object to be built on with each contour
    #f: Placefile, edited throughout the program
    return zdr_storm_lon,zdr_storm_lat,zdr_dist,zdr_forw,zdr_back,zdr_areas,zdr_centroid_lon,zdr_centroid_lat,zdr_mean,zdr_cc_mean,zdr_max,zdr_masks,zdr_outlines,ax,f