import numpy as np
import numpy.ma as ma
from shapely import geometry
from shapely.ops import transform
from metpy.units import check_units, concatenate, units
from matplotlib.path import Path
from pyproj import Geod

def zhh_objects(zhhc,REFmasked,rlons,rlats,max_lons_c,max_lats_c,proj):
    zhh_areas = []
    zhh_centroid_lon = []
    zhh_centroid_lat = []
    zhh_max = []
    zhh_storm_lon = []
    zhh_storm_lat = []
    zhh_core_avg = []
    if np.max(REFmasked) > 35.0:
        for level in zhhc.collections:
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
                        
                try:
                    pr_area = (transform(proj, polygon).area * units('m^2')).to('km^2')
                    boundary = np.asarray(polygon.boundary.xy)
                    polypath = Path(boundary.transpose())
                    coord_map = np.vstack((rlons[0,:,:].flatten(), rlats[0,:,:].flatten())).T # create an Mx2 array listing all the coordinates in field
                    mask_zhh = polypath.contains_points(coord_map).reshape(rlons[0,:,:].shape)
                    storm_core = REFmasked[mask_zhh]
                    np.savetxt("storm_core_pre.csv",storm_core,delimiter=",")
                    if pr_area > 2 * units('km^2') and np.max(REFmasked[mask_zhh]) > 35.0:
                        storm_core = ma.masked_where(storm_core < np.percentile(storm_core,95), storm_core)
                        core_zhh_mean = np.mean(storm_core)
                        g = Geod(ellps='sphere')
                        dist_zhh = np.zeros((np.asarray(max_lons_c).shape[0]))
                        rlons_mask = rlons[0,mask_zhh]
                        rlats_mask = rlats[0,mask_zhh]
                        for i in range(max_lons_c.shape[0]):
                            distance_zhh = np.zeros((np.asarray(rlons_mask).shape[0]))
                            for j in range(rlons_mask.shape[0]):
                                dist_zhh = g.inv(rlons_mask[j], rlats_mask[j], max_lons_c[i], max_lats_c[i])
                                distance_zhh[j] = dist_zhh[2]/1000.0

                            if np.min(distance_zhh)<0.5:
                                zhh_areas.append((pr_area))
                                zhh_centroid_lon.append((polygon.centroid.x))
                                zhh_centroid_lat.append((polygon.centroid.y))
                                zhh_storm_lon.append(max_lons_c[i])
                                zhh_storm_lat.append(max_lats_c[i])
                                zhh_max.append((np.max(REFmasked[mask_zhh])))
                                zhh_core_avg.append((core_zhh_mean))
                except:
                    print('reflectivity error')
    return zhh_areas,zhh_centroid_lon,zhh_centroid_lat,zhh_storm_lon,zhh_storm_lat,zhh_max,zhh_core_avg