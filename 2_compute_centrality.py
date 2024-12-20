# This script automates computation of several centrality variants. Please consult README.md for further information.

import algo.net_helper as nh
import os
import geopandas as gpd
from timeit import default_timer as timer
import numpy as np
import algo.centrality as centr
from datetime import datetime as dt
import platform
import psutil

# settings

# aoi_name (the "case_id" property of NetAScore; used as part of file names)
aoi_name = "at_zs"

# if recalc_existing, then previously computed centrality results will be re-computed and overwritten
# if False, skips already computed centrality variants and appends newly computed columns to output file
recalc_existing = False

# settings for network simplification
simplify_net = {
    "remove_ends": True,
    "merge_segments": True,
    "seg_always_merge_zero_w_nodes": True, 
    "seg_merge_threshold": 250
}

# distances to use for node subsampling (approximate regular grid)
# None means compute results for full network without subsampling
nodes_weight_grid_sample_dist = [None, 300, 600, 900, 1200, 1500]
# this parameter sets the number of random nodes used as origins for length-unrestricted standard betweenness centrality
ebc_random_sample_n = 5000  # option to approximate std ebc (using random node subsample of size n)

# centrality settings (variants to compute ["sp", "bp"]: shortest and/or bikeable paths)
compute_ebc_for = []        # standard edge betweenness centrality
nws_ebc = False
compute_febc_for = []       # spatially filtered standard edge betweenness centrality (excludes nodes with spatial weight == 0)
nws_febc = False
compute_sbc_for = ["bp", "sp"]  # spatial betweenness centrality
nws_sbc = True
compute_pbc_for = []        # population-weighted betweenness centrality
nws_pbc = False

bp_detour_factors = [4]       # detour factors to use for bikeable path centrality

### distance cutoff values for each centrality type // 0: no cutoff (compute for all nodes, all possible routes)
ebc_dist_cutoffs = [None] #2000, 4000, 7000]
ebc_dist_decays = None 
febc_dist_cutoffs = [2000, 4000, 7000, 4000, 7000, 7000] 
febc_dist_decays = [None, None, None, 2000, 4000, 2000]     
sbc_dist_cutoffs = [2000, 4000, 7000, 4000, 7000, 7000] 
sbc_dist_decays = [None, None, None, 2000, 4000, 2000]
pbc_dist_cutoffs = [2000, 4000, 7000] 
pbc_dist_decays = None

mode = "bike_incwalk"
# tolerable access is determined by input network: all segments that have an index value assigned 
# (other than NULL, > 0) but have mode access set to False
access = "bicycle"
access_penalty_factor = 4 # currently will always be 4 - TODO: fix!

dir_data = "data"
dir_data_in = "data_input"

# settings for parallelization
processes = None # None for no. of cores
chunksize = 250   # e.g. 10 - 250
tasks_per_child = 50  # None for default value (unlimited)


def compute_centrality(path_type = "bp", centrality_type="sbc", bp_detour_factor=4, access_penalty_factor=4, out_gdf=None, f_out=None, nw_grid_sample_dist=None, random_sample_n=None):
    # computed params
    dist_cutoffs = sbc_dist_cutoffs
    dist_decays = sbc_dist_decays
    node_weight_col = "w_spatial"
    routing_weight_col = f"cost_{mode}_ft"
    weighted_centrality = True
    normalized = True
    od_w_norm_ref = None
    use_pop_weight = False
    if centrality_type == "ebc":
        weighted_centrality = False
        dist_cutoffs = ebc_dist_cutoffs
        dist_decays = ebc_dist_decays
        node_weight_col = "w_uniform"
    elif centrality_type == "febc":
        weighted_centrality = True
        dist_cutoffs = febc_dist_cutoffs
        dist_decays = febc_dist_decays
        node_weight_col = "w_uniform_filtered"
    elif centrality_type == "pbc":
        dist_cutoffs = pbc_dist_cutoffs
        dist_decays = pbc_dist_decays
        node_weight_col = "w_pop"
        normalized = False
        use_pop_weight = True
    elif centrality_type != "sbc":
        raise Exception(f"The specified centrality_type '{centrality_type}' is unknown. Choose ebc/sbc/pbc")
    if path_type == "sp":
        routing_weight_col = "length"
    # generated params
    f_network = os.path.join(dir_data_in, f"netascore_{aoi_name}.gpkg")
    f_nodes_weight = None

    # determine whether all requested centrality variants already exist - skip before network loading and prep.
    if not recalc_existing and not f_out is None and os.path.isfile(f_out):
        n_exist = 0
        i = 0
        for dist_cutoff in dist_cutoffs:
            dist_decay = None
            if dist_cutoff is None or dist_cutoff < 1:
                dist_cutoff = 0
            elif dist_decays and len(dist_decays) > i:
                dist_decay = dist_decays[i]
            if dist_decay:
                centr_name = f"{centrality_type}_dec_{dist_decay}_{dist_cutoff}_{path_type}"
            else:
                centr_name = f"{centrality_type}_c{dist_cutoff}_{path_type}"
            if path_type == "bp":
                centr_name += f"_d{bp_detour_factor}"
            # check whether this centrality already exists in output file
            if nw_grid_sample_dist is not None:
                centr_name += f"_nws_{nw_grid_sample_dist}"
            if f"centr_{centr_name}_sum" in nh.read_cols_from_gpkg(f_out):
                print(f"centrality '{centr_name}' already exists")
                n_exist +=1
            i+=1
        if n_exist == len(dist_cutoffs):
            print("All requested centrality variants for this run exist. Skipping.")
            return

    # load filtered network (connected, only accessible to mode)
    print("loading and preparing network")
    edges, nodes, g = nh.get_net_from_file(f_network, mode, access, bp_detour_factor, access_penalty_factor)
    # output
    if out_gdf is None: #and (not os.path.isfile(f_out)):
        out_gdf = edges
    
    # determine node weights
    nodes_weight = None
    if weighted_centrality:
        f_nodes_weight = os.path.join(dir_data, f"nodes_weight_{aoi_name}.gpkg")
        # load node weights
        print("loading node weights")
        nodes_weight = gpd.read_file(f_nodes_weight, engine='pyogrio', fid_as_index=True)
        if centrality_type == "febc":
            # append uniform_filtered weight col
            nodes_weight = nodes_weight.assign(w_uniform_filtered=0.0)
            nodes_weight.loc[nodes_weight["w_spatial"]>0, "w_uniform_filtered"] = 1.0
    else:
        # prepare node weights gdf with weight=1
        nodes_weight = nodes.assign(w_uniform=1.0)
    
    # check whether node weight column is available
    if not node_weight_col in nodes_weight.columns:
        if node_weight_col == "w_pop":
            raise Exception(f"ERROR: Population node weight column not available ({node_weight_col}). Please first compute node weights with 'generate_population_weights = True', or remove pbc variants from the settings ('compute_pbc_for = []').")
        raise Exception(f"ERROR: Node weight column not available ({node_weight_col}). Please change settings or compute weights first.")

    # simplification of network
    if simplify_net:
        _n_before = g.number_of_nodes()
        _e_before = g.number_of_edges()
        nh.net_simplify(g, nodes_weight, node_weight_col, routing_weight_col, 
                        remove_ends=simplify_net["remove_ends"], merge_segments=simplify_net["merge_segments"],
                        seg_always_merge_zero_w_nodes=simplify_net["seg_always_merge_zero_w_nodes"], seg_merge_threshold=simplify_net["seg_merge_threshold"])
        # log output
        _n_after = g.number_of_nodes()
        _e_after = g.number_of_edges()
        with open(f_out.replace(".gpkg", ".txt"), "a") as logfile:
            logfile.write(f"""
        {dt.now()}: Simplifying network with settings: {simplify_net}
        nodes: {_n_before} -> {_n_after} ({(_n_before-_n_after)/_n_before:.2%})
        edges: {_e_before} -> {_e_after} ({(_e_before-_e_after)/_e_before:.2%})
        """)

    # sample node weights based on regular grid
    _name_str_nws_ = ""
    if nw_grid_sample_dist is not None and nw_grid_sample_dist > 0:
        nodes_weight = nh.nodes_weight_grid_sample(nodes_weight, node_weight_col, g, nw_grid_sample_dist)
        _name_str_nws_ = f"_nws_{nw_grid_sample_dist}"
    # iterate: distance cutoff values
    i = 0
    for dist_cutoff in dist_cutoffs:
        dist_decay = None
        if dist_cutoff is None or dist_cutoff < 1:
            dist_cutoff = 0
        elif dist_decays and len(dist_decays) > i:
            dist_decay = dist_decays[i]
        if dist_decay:
            centr_name = f"{centrality_type}_dec_{dist_decay}_{dist_cutoff}_{path_type}"
        else:
           centr_name = f"{centrality_type}_c{dist_cutoff}_{path_type}"
        if path_type == "bp":
            centr_name += f"_d{bp_detour_factor}"
        centr_name += _name_str_nws_

        # check whether this centrality already exists in the output file
        if not f_out is None and os.path.isfile(f_out) and f"centr_{centr_name}_sum" in nh.read_cols_from_gpkg(f_out):
            print(f"centrality '{centr_name}' already exists")
            if recalc_existing:
                print("...recalculating it now.")
            else:
                print("...skipping.")
                i+=1
                continue

        # compute weight reference for spatial and population-weighted centrality
        if centrality_type in ["pbc", "sbc"]:
            od_w_norm_ref=dist_cutoff**2 * np.pi
            if use_pop_weight:
                od_w_norm_ref *= 0.004 # use reference value (population density) for normalization

        # centrality computation
        print(f"\nStarting centrality computation: {centr_name}")
        tstart = timer()
        if dist_cutoff < 1:
            dist_cutoff = None
        if od_w_norm_ref == 0:
            od_w_norm_ref = None
        c = centr.spatial_betweenness_centrality(g, nodes_weight, weight_col=routing_weight_col, normalized=normalized, 
                                    dist_cutoff=dist_cutoff, od_w_norm_ref=od_w_norm_ref, node_weight_col=node_weight_col, 
                                    processes=processes, chunksize=chunksize, tasks_per_child=tasks_per_child, dist_decay_from=dist_decay, random_sample_n=random_sample_n) 
                                    # will use net_dist_filter = dist_cutoff internally 
        if f"centr_{centr_name}_sum" in out_gdf.columns:
            out_gdf.drop(columns=[f"centr_{centr_name}_ft", f"centr_{centr_name}_tf", f"centr_{centr_name}_sum"], inplace=True)
        nh.add_centr_to_netascore(out_gdf, nh.centr_add_orig_edges(c) if simplify_net else c, centr_name, "edge_id")
        tend = timer()
        #print("--- Centrality computation TOTAL TIME:", tend-tstart, "---")
        with open(f_out.replace(".gpkg", ".txt"), "a") as logfile:
            logfile.write(f"""
        {dt.now()}: finished computing '{centr_name}' after {tend-tstart:.1f} seconds ({(tend-tstart)/60:.1f} min / {(tend-tstart)/3600:.2f} hrs)
        (chunksize: {chunksize}, max. tasks_per_child: {tasks_per_child}, processes: {processes})
        routing weight column: {routing_weight_col}
        node weight column:    {node_weight_col}
        normalized:            {normalized}
        od_w_norm_ref:         {od_w_norm_ref}
        dist_cutoff:           {dist_cutoff}
        dist_decay:            {dist_decay}
        access_penalty_factor: {access_penalty_factor}
        n_w_grid_sample_dist:  {nw_grid_sample_dist}
            """)

        print("...joined centrality results to NetAScore GeoDataFrame.")
        print(f"-> finished computing '{centr_name}' after {tend-tstart:.1f} seconds ({(tend-tstart)/60:.1f} min / {(tend-tstart)/3600:.2f} hrs)")

        if os.path.isfile(f_out):
            # Append centrality result columns to result file
            print(f"...appending centrality results for 'centr_{centr_name}' to output file '{f_out}'")
            nh.add_col_to_gpkg(f_out, out_gdf[f"centr_{centr_name}_ft"])
            nh.add_col_to_gpkg(f_out, out_gdf[f"centr_{centr_name}_tf"])
            nh.add_col_to_gpkg(f_out, out_gdf[f"centr_{centr_name}_sum"])
        else:
            # create output geopackage
            print("Creating output file:", f_out)
            out_gdf.to_file(f_out)
        
        print(f"Saved results to file '{f_out}'.")
        i+=1
    return out_gdf

def load_output_gdf(file_path):
    gdf = gpd.read_file(file_path)
    #if input does not have 'edge_id' column, generate it from index
    if not 'edge_id' in gdf.columns:
        gdf['edge_id'] = gdf.index
        print("net helper: created 'edge_id' column from gdf index.")
    return gdf

if __name__ == '__main__':
    # create directories
    if not os.path.exists(dir_data):
        os.makedirs(dir_data)
    # if output file already exists, handle (skip/overwrite) existing cols and append new centrality measures
    f_out = os.path.join(dir_data, f"r_{aoi_name}_edges.gpkg")
    if os.path.isfile(f_out):
        print("Output file already exists. Will", "recalc and overwrite" if recalc_existing else "skip", "existing columns.")
    with open(f_out.replace(".gpkg", ".txt"), "a") as logfile:
        logfile.write(f"""
    ----------
    {dt.now()}: Running centrality computation script on: 
            {platform.uname()}
            --- user: '{os.getlogin()}', dir: '{os.getcwd()}'
            --- Python version {platform.python_version()}, {platform.python_implementation()} ---
            --- CPU: {psutil.cpu_count()} cores ({psutil.cpu_count(logical=False)} physical), freq: {psutil.cpu_freq()}
            --- RAM stats: {psutil.virtual_memory()}
        compute_ebc_for:   {compute_ebc_for}
        compute_sbc_for:   {compute_sbc_for}
        compute_pbc_for:   {compute_pbc_for}
        bp_detour_factors: {bp_detour_factors}
        ebc_dist_cutoffs:  {ebc_dist_cutoffs}
        sbc_dist_cutoffs:  {sbc_dist_cutoffs}
        pbc_dist_cutoffs:  {pbc_dist_cutoffs}
        n_w_grid_sample_d:   {nodes_weight_grid_sample_dist}
        """)
    print("starting centrality computation...")
    
    for nw_grid_sample_d in nodes_weight_grid_sample_dist:
        ### spatial betweenness centrality 
        if nws_sbc or nw_grid_sample_d is None:
            # bikeable path sbc
            if compute_sbc_for and "bp" in compute_sbc_for:
                for detour_factor in bp_detour_factors:
                    compute_centrality(path_type="bp", bp_detour_factor=detour_factor, access_penalty_factor=access_penalty_factor, f_out=f_out, nw_grid_sample_dist=nw_grid_sample_d)
            # shortest path sbc
            if compute_sbc_for and "sp" in compute_sbc_for:
                compute_centrality(path_type="sp", f_out=f_out, nw_grid_sample_dist=nw_grid_sample_d)
        ### population-weighted bc
        if nws_pbc or nw_grid_sample_d is None:
            # bikeable path pbc
            if compute_pbc_for and "bp" in compute_pbc_for:
                for detour_factor in bp_detour_factors:
                    compute_centrality(path_type="bp", centrality_type="pbc", bp_detour_factor=detour_factor, access_penalty_factor=access_penalty_factor, f_out=f_out, nw_grid_sample_dist=nw_grid_sample_d)
            # shortest path pbc
            if compute_pbc_for and "sp" in compute_pbc_for:
                compute_centrality(path_type="sp", centrality_type="pbc", f_out=f_out, nw_grid_sample_dist=nw_grid_sample_d)
        ### standard edge betweenness centrality
        # skip standard ebc if node weight sampling dist is specified
        if nws_ebc or nw_grid_sample_d is None:
            # bikeable path ebc
            if compute_ebc_for and "bp" in compute_ebc_for:
                for detour_factor in bp_detour_factors:
                    compute_centrality(path_type="bp", centrality_type="ebc", bp_detour_factor=detour_factor, access_penalty_factor=access_penalty_factor, f_out=f_out, nw_grid_sample_dist=nw_grid_sample_d, random_sample_n=ebc_random_sample_n)
            # shortest path ebc
            if compute_ebc_for and "sp" in compute_ebc_for:
                compute_centrality(path_type="sp", centrality_type="ebc", f_out=f_out, nw_grid_sample_dist=nw_grid_sample_d, random_sample_n=ebc_random_sample_n)
            # shortest path febc
        # bikeable path febc
        if nws_febc or nw_grid_sample_d is None:
            if compute_febc_for and "bp" in compute_febc_for:
                for detour_factor in bp_detour_factors:
                    compute_centrality(path_type="bp", centrality_type="febc", bp_detour_factor=detour_factor, access_penalty_factor=access_penalty_factor, f_out=f_out, nw_grid_sample_dist=nw_grid_sample_d)
            # shortest path febc
            if compute_febc_for and "sp" in compute_febc_for:
                compute_centrality(path_type="sp", centrality_type="febc", f_out=f_out, nw_grid_sample_dist=nw_grid_sample_d)
        