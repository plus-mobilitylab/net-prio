# This script computes OD node weights which are required as input for centrality computation.

import algo.net_helper as nh
import os.path
import geopandas as gpd

# settings
aoi_name = "at_zs"
replace_existing = False

# optional: if population-weighted centrality should be computed in later stages,
# enable generation of population weight column here
# NOTE: this requires population raster files covering (each of) your area(s) of interest
# (GeoTIFF) in the subdirectory data_input/population_raster/ (will find the spatially matching
# ones for you if several files with different extent are provided). Currently, each area of 
# interest needs to be fully covered by a single GeoTIFF (please merge before if needed)
generate_population_weights = False

dir_data = "data"
dir_data_in = "data_input"

mode = "bike_incwalk"
# tolerable access is determined by input network: all segments that have an index value assigned 
# (other than NULL, > 0) but have mode access set to False
access = "bicycle" 

display_large_maps = True

# segment distance value used by momepy tessellation algorithm: increase if running out of memory
TESS_SEG_DIST = 0.5

# generated params
f_network = os.path.join(dir_data_in, f"netascore_{aoi_name}.gpkg")
f_osm_poly = os.path.join(dir_data, f"osm_poly_{aoi_name}.gpkg")
f_tessellation = os.path.join(dir_data, f"tessellation_{aoi_name}.gpkg")
f_nodes_weight = os.path.join(dir_data, f"nodes_weight_{aoi_name}.gpkg")
# create directories
if not os.path.exists(dir_data):
    os.makedirs(dir_data)
if not os.path.exists(os.path.join("plots", "svg")):
    os.makedirs(os.path.join("plots", "svg"))

# load filtered network (connected, only accessible to mode)
edges, nodes, g = nh.get_net_from_file(f_network, mode, access)

# load polygons for filtering spatial weights (urban areas etc.)
aoi = nh.get_aoi(edges, buffer=100)
bbox = nh.transform_geom(aoi, edges.crs).bounds
nh.get_osm_data_qfile("overpass_area_query.txt", bbox, f_osm_poly, replace_existing)
polys = gpd.read_file(f_osm_poly, layer="multipolygons", columns=["osm_id"])
dest_poly = nh.transform_geom(polys.union_all(), 4326, edges.crs, inv_xy=True)
#display(dest_poly)
obj = {
    "type": ["urban"],
    "geometry": [dest_poly],
    "weight_factor": [1]
}
poly_df = gpd.GeoDataFrame.from_dict(obj, crs=edges.crs)

# tessellation and weight polygons
if replace_existing or not os.path.exists(f_tessellation):
    tess = nh.tessellate(edges, limit=aoi, out_file=f_tessellation, segment_dist=TESS_SEG_DIST)
else:
    tess = gpd.read_file(f_tessellation)

weight_polygons = gpd.overlay(tess, poly_df)
weight_polygons["area"] = weight_polygons.geometry.area
weight_polygons.set_index("edge_id", inplace=True)

# plotting result
import matplotlib.pyplot as plt
print(f"Plotting intersected tessellation result")
plt.rcParams["figure.figsize"] = (20,20)
ax = weight_polygons.boundary.plot(edgecolor="red", lw=0.5)
ax = edges.plot(ax=ax, lw=0.3)
ax = nodes.plot(ax=ax, markersize=0.1)
plt.margins(0)
plt.axis('off')
plt.savefig(fname=f"plots/tessel_{aoi_name}.pdf", bbox_inches='tight')

nodes_weight = nh.add_node_weights(nodes, edges, weight_polygons, weight_col="area", 
                output_col="w_spatial",
                output_file= None if generate_population_weights else f_nodes_weight)

if generate_population_weights:
    # alternative weighting approach: use population data instead of spatial-weighted with polygon filter
    dir_popraster = os.path.join(dir_data_in, "population_raster") 
    tess = nh.add_population(tess, dir_popraster, out_file=os.path.join(dir_data, f"tess_pop_{aoi_name}.gpkg"))
    nodes_weight = nh.add_node_weights(nodes_weight, edges, tess.set_index("edge_id"), 
                                       weight_col="population", output_col="w_pop",
                                       output_file=f_nodes_weight)