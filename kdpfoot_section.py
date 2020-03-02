import numpy as np
from shapely import geometry
from shapely.ops import transform
from metpy.units import atleast_1d, check_units, concatenate, units
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from pyproj import Geod

def kdp_objects(kdpc,KDPmasked,ax,f,time_start,month,d_beg,h_beg,min_beg,sec_beg,d_end,h_end,min_end,sec_end,rlons,rlats,max_lons_c,max_lats_c,kdplev,proj):
    kdp_areas = []
    kdp_centroid_lon = []
    kdp_centroid_lat = []
    kdp_max = []
    kdp_storm_lon = []
    kdp_storm_lat = []
    if np.max(KDPmasked) > kdplev:
        for level in kdpc.collections:
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
                except:
                    continue
                boundary = np.asarray(polygon.boundary.xy)
                polypath = Path(boundary.transpose())
                coord_map = np.vstack((rlons[0,:,:].flatten(), rlats[0,:,:].flatten())).T # create an Mx2 array listing all the coordinates in field
                mask_kdp = polypath.contains_points(coord_map).reshape(rlons[0,:,:].shape)
                if pr_area > 2 * units('km^2'):
                    g = Geod(ellps='sphere')
                    dist_kdp = np.zeros((np.asarray(max_lons_c).shape[0]))
                    for i in range(dist_kdp.shape[0]):
                                distance_kdp = g.inv(polygon.centroid.x, polygon.centroid.y,
                                                       max_lons_c[i], max_lats_c[i])
                                dist_kdp[i] = distance_kdp[2]/1000.

                    if np.min(np.asarray(dist_kdp)) < 15.0:
                        kdp_path = polypath
                        kdp_areas.append((pr_area))
                        kdp_centroid_lon.append((polygon.centroid.x))
                        kdp_centroid_lat.append((polygon.centroid.y))
                        kdp_storm_lon.append((max_lons_c[np.where(dist_kdp == np.min(dist_kdp))[0][0]]))
                        kdp_storm_lat.append((max_lats_c[np.where(dist_kdp == np.min(dist_kdp))[0][0]]))
                        kdp_max.append((np.max(KDPmasked[mask_kdp])))
                        patch = PathPatch(polypath, facecolor='green', alpha=.5, edgecolor = 'green', linewidth = 3)
                        ax.add_patch(patch)
                        #Add polygon to placefile
                        f.write('TimeRange: '+str(time_start.year)+'-'+str(month)+'-'+str(d_beg)+'T'+str(h_beg)+':'+str(min_beg)+':'+str(sec_beg)+'Z '+str(time_start.year)+'-'+str(month)+'-'+str(d_end)+'T'+str(h_end)+':'+str(min_end)+':'+str(sec_end)+'Z')
                        f.write('\n')
                        f.write("Color: 000 139 000 \n")
                        f.write('Line: 3, 0, "KDP Foot Outline" \n')
                        for i in range(len(kdp_path.vertices)):
                            f.write("%.5f" %(kdp_path.vertices[i][1]))
                            f.write(", ")
                            f.write("%.5f" %(kdp_path.vertices[i][0]))
                            f.write('\n')
                        f.write("End: \n \n")
    return kdp_areas,kdp_centroid_lon,kdp_centroid_lat,kdp_storm_lon,kdp_storm_lat,kdp_max,ax,f