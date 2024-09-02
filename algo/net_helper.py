import geopandas as gpd
import pandas as pd
import os
import momepy as mp
import pyproj
import shapely as sly
import subprocess
from urllib.error import HTTPError
import urllib.request
from osgeo import ogr
import os.path
import networkx as nx
import numpy as np
from pathlib import Path
import rasterio as rio
import rasterstats

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

os.environ['USE_PYGEOS'] = '0'


cols_static = ["edge_id", "osm_id", "length", "geometry"]
cols_dir_basic = ["access_bicycle", "access_pedestrian"]

overpass_api_endpoints = [
    "https://overpass-api.de/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter"
]

# TODO: check whether OK to remove overpass_query_string
overpass_query_string = ""
if os.path.exists('overpass_area_query.txt'):
    with open('overpass_area_query.txt', 'r', encoding="utf-8") as f:
        overpass_query_string = f.read()
        
ea_path = {}
ea_ids = {"cur":-1, "start":-1}

def netascore_generate_subset(dir, in_file, out_file, clip_geom):
    tmp = gpd.read_file(os.path.join(dir, in_file), layer="edge", rows=10)
    srid = tmp.crs
    cg = clip_geom.to_crs(srid).envelope
    print(cg)
    edges = gpd.read_file(os.path.join(dir, in_file), layer="edge", mask=cg)
    edges.to_file(os.path.join(dir, out_file), layer="edge", driver="GPKG")
    # temp workaround: add "node_id" column to maintain original node id after subsetting
    nodes = gpd.read_file(os.path.join(dir, in_file), layer="node")
    nodes["node_id"] = nodes.index + 1 # add node_id column
    nodes.to_file(os.path.join(dir, in_file + "_nid"), layer="node", driver="GPKG")
    nodes = gpd.read_file(os.path.join(dir, in_file + "_nid"), layer="node", mask=cg)
    nodes.to_file(os.path.join(dir, out_file), layer="node", driver="GPKG")
    return cg

def netascore_to_routable_net(netascore_gdf: gpd.GeoDataFrame, mode="bike", access="bicycle", routing_factor = 4, access_penalty_factor = 4):
    # if input does not have 'edge_id' column, generate it from index
    if not 'edge_id' in netascore_gdf.columns:
        netascore_gdf['edge_id'] = netascore_gdf.index
        print("net helper: created 'edge_id' column from gdf index.")
    cols_dir = cols_dir_basic + [f"index_{mode}"]
    # filter input df
    cols = cols_static + ["from_node", "to_node"] + [f"{x}_ft" for x in cols_dir] + [f"{x}_tf" for x in cols_dir]
    net_a = netascore_gdf.filter(cols, axis=1).copy()
    # append inverted state
    net_a["inv"] = False
    # generate mapping for renaming dir-specific columns
    mapping = {f'{k}_ft': f'{k}_tf' for k in cols_dir}
    mapping.update({f'{k}_tf': f'{k}_ft' for k in cols_dir})
    mapping.update({"from_node":"to_node", "to_node":"from_node"})
    net_b = net_a.rename(columns=mapping)
    net_b["inv"] = True
    # append inverted direction net
    net = None
    net = pd.concat([net_a, net_b], ignore_index=True)
    # remove inverted-dir columns
    net.drop([f'{k}_tf' for k in cols_dir], axis=1, inplace=True, errors="ignore")
    # append routing cost columns
    print(f"orig: {len(netascore_gdf)}, routable (di-): {len(net)}")
    net[f"cost_{mode}_ft"] = ((1 + (1-net[f"index_{mode}_ft"]) * routing_factor) * net['length'])
    # apply penalty for segments with non-legal access (e.g. pushing bike for short
    # pedestrian-only section) if no alternative available / high detour induced
    net.loc[~net[f"access_{access}_ft"], f"cost_{mode}_ft"] *= access_penalty_factor
    # FUTURE: set a minimum cost for such segments
    # also treat stairs differently: apply additional, high penalty factor (e.g. 50 * length -> 10 m stairs is 500 m optimum bikeability without stairs)
    return net

def get_joined_centr_output(netascore_gdf: gpd.GeoDataFrame, centrality, centr_name: str, edge_key: str = "edge_id"):
    # convert from dict with compound key to pandas df
    print("converting results to gdf...")
    tdf = pd.DataFrame(list(centrality.keys()))
    tdf.rename(columns={0:"from_node", 1:"to_node", 2:"edge_key"}, inplace=True)
    print("assigning centrality values...")
    tdf["centrality"] = centrality.values()
    # map centrality value back to original (geo)pandas df (for both directions)
    print("merging centrality to original gdf...")
    if netascore_gdf.index.name == edge_key and edge_key in netascore_gdf.columns:
        netascore_gdf.index.name = "index"
    net_tmp = netascore_gdf.merge(tdf, left_on=["from_node", "to_node", edge_key], right_on=["from_node", "to_node", "edge_key"], how="left", suffixes=[None, "_b"])
    net_tmp.rename(columns={"centrality":"centrality_ft"}, inplace=True)
    net_ready = net_tmp.merge(tdf, left_on=["to_node", "from_node", "edge_key"], right_on=["from_node", "to_node", "edge_key"], how="left", suffixes=[None, "_c"])
    net_ready.rename(columns={"centrality":"centrality_tf"}, inplace=True)
    net_ready.set_index("edge_key", drop=False, inplace=True)
    print("...done merging.")
    return net_ready

def add_centr_to_netascore(netascore_gdf: gpd.GeoDataFrame, centrality, centr_name: str, edge_key: str = "edge_id"):
    print("adding centrality results to NetAScore GeoDataFrame...")
    net_ready = get_joined_centr_output(netascore_gdf, centrality, centr_name, edge_key)
    if edge_key in netascore_gdf.columns:
        netascore_gdf.set_index(edge_key, drop=False, inplace=True)
        netascore_gdf.index.name = "index"
    if "index" in netascore_gdf.columns:
        netascore_gdf.drop(columns=["index"], inplace=True)
    print("joining with NetAScore gdf...")
    netascore_gdf[f"centr_{centr_name}_ft"] = net_ready[~net_ready.index.isnull()].centrality_ft
    netascore_gdf[f"centr_{centr_name}_tf"] = net_ready[~net_ready.index.isnull()].centrality_tf
    centr_sum = net_ready.centrality_tf + net_ready.centrality_ft
    netascore_gdf[f"centr_{centr_name}_sum"] = centr_sum[~centr_sum.index.isnull()]
    print("...done joining.")

def add_col_to_gpkg(out_file:str, col:pd.Series, table_name:str = None, target_idx="edge_id"):
    print("appending data to:", out_file)
    import sqlite3
    import pandas as pd
    con = sqlite3.connect(out_file)
    # import to temp table
    col.to_sql("tmp_import", con, if_exists="replace", index=True, index_label="idx")
    # add column to target table
    # first, prepare for altering GPKG
    con.enable_load_extension(True)
    con.execute("""SELECT load_extension("mod_spatialite");""")
    # determine table name if not explicitly specified
    if table_name is None:
        table_name = con.execute("SELECT table_name FROM gpkg_contents LIMIT 1").fetchone()[0]
    # check if column already exists
    result = con.execute(f"SELECT name FROM PRAGMA_TABLE_INFO('{table_name}');").fetchall()
    if col.name in [r[0] for r in result]:
        # if so, drop the column
        con.execute(f"ALTER TABLE {table_name} DROP COLUMN {col.name};")
    con.execute(f"ALTER table {table_name} ADD COLUMN {col.name} REAL;")
    # append data to table
    con.execute(f"""
    UPDATE {table_name}
    SET {col.name} = tmp_import.{col.name}
    FROM tmp_import
    WHERE tmp_import.idx = {table_name}.{target_idx};
    """)
    # drop temp table, commit and close connection
    con.execute(f"DROP TABLE tmp_import;")
    con.commit()
    con.close()
    print("Done appending data to GeoPackage.")
    
def read_cols_from_gpkg(gpkg_file, table_name=None):
    import sqlite3
    import pandas as pd
    con = sqlite3.connect(gpkg_file)
    if table_name is None:
        table_name = con.execute("SELECT table_name FROM gpkg_contents LIMIT 1").fetchone()[0]
    result = con.execute(f"SELECT name FROM PRAGMA_TABLE_INFO('{table_name}');").fetchall()
    con.close()
    return [r[0] for r in result]

def delete_cols_from_gpkg(gpkg_file, drop_cols, table_name=None, use_copy=True):
    import sqlite3
    con = sqlite3.connect(gpkg_file)
    if table_name is None:
        table_name = con.execute("SELECT table_name FROM gpkg_contents LIMIT 1").fetchone()[0]
    # first, prepare for altering GPKG
    con.enable_load_extension(True)
    con.execute("""SELECT load_extension("mod_spatialite");""")
        
    if use_copy:
        # get available cols
        result = con.execute(f"SELECT name FROM PRAGMA_TABLE_INFO('{table_name}');").fetchall()
        cols = [r[0] for r in result]
        keep_cols = []
        for col in cols:
            # check if column exists
            if not col in drop_cols:
                keep_cols.append(col)
        # crete new table with remaining cols
        colstr = ','.join([f'"{c}"' for c in keep_cols])
        print(colstr)
        con.execute(f"CREATE TABLE _dc_temp_ AS SELECT {colstr} FROM {table_name};")
        # drop original table
        con.execute(f"DROP TABLE {table_name};")
        # rename temp table
        con.execute(f"ALTER TABLE _dc_temp_ RENAME TO {table_name};")
    else:
        # get available cols
        result = con.execute(f"SELECT name FROM PRAGMA_TABLE_INFO('{table_name}');").fetchall()
        cols = [r[0] for r in result]
        for dc in drop_cols:
            # check if column exists
            if dc in cols:
                # if so, drop the column
                con.execute(f"ALTER TABLE {table_name} DROP COLUMN {dc};")
                print("dropped column:", dc)
            else:
                print("column not available for dropping:", dc)
    # free space
    con.execute("VACUUM;")
    con.commit()
    con.close()

def verify_net_ids(edges, nodes):
    # check that node ids match edge FKs
    ns = nodes.sample()
    n_id = ns.iloc[0].name
    print(n_id)
    map = nodes[nodes.index == n_id].explore()
    edges[(edges.from_node == n_id) | (edges.to_node == n_id)][["geometry", "osm_id", "from_node", "to_node"]].explore(m=map)
    return map

def get_net_from_file(f_network, mode, access, routing_factor=4, access_penalty_factor=4):
    # import network
    edges_all = gpd.read_file(f_network, layer="edge")
    # INFO: reading fid does not work with fiona in geopandas: see https://github.com/geopandas/geopandas/issues/1035
    # and: https://github.com/geopandas/geopandas/issues/2794 (with hint to use pyogrio instead)
    nodes_all = gpd.read_file(f_network, engine='pyogrio', fid_as_index=True, layer="node")
    net_routing = netascore_to_routable_net(edges_all, mode=mode, access=access, routing_factor=routing_factor, access_penalty_factor=access_penalty_factor)
    # filter net_routing for largest connected, bike-accessible sub-network
    net_routing = net_routing[~net_routing[f"index_{mode}_ft"].isna()]
    # generate NetworkX graph
    g = nx.from_pandas_edgelist(net_routing, source='from_node', target='to_node', 
                                edge_attr=True, create_using=nx.MultiDiGraph, edge_key="edge_id")
    # get largest connected component (using weakly_connected_components, as this is implemented for digraphs)
    n_acc_g = max(nx.weakly_connected_components(g), key=len)
    # filter network to accessible sub-net
    
    print(f"{len(n_acc_g)} out of {len(nodes_all)} nodes are accessible within the largest connected component ({len(n_acc_g)/len(nodes_all):.1%})")
    # determine nodes with tolerable access (with routing penalty applied)
    c = net_routing[net_routing[f"access_{access}_ft"]].from_node.unique()
    d = net_routing[net_routing[f"access_{access}_ft"]].to_node.unique()
    acc_tol_nodes = nodes_all[~nodes_all.index.isin(np.unique(np.append(c,d)))]
    acc_tol_nodes = acc_tol_nodes[acc_tol_nodes.index.isin(n_acc_g)]
    nodes_all["acc_strict"] = False
    nodes_all.loc[(nodes_all.index.isin(n_acc_g) & ~nodes_all.index.isin(acc_tol_nodes.index)), "acc_strict"] = True
    c_nodes = nodes_all[nodes_all.index.isin(n_acc_g)]
    #if display_large_maps:
    #    print("non-accessible nodes")
    #    display(nodes_all[~nodes_all.index.isin(n_acc_g)].explore())
    #    #print("accessible nodes")
    #    #display(nodes_acc_strict.explore())
    #    print("tolerable access nodes")
    #    display(acc_tol_nodes.explore())
    # filter network to largest connected component
    cg = g.subgraph(n_acc_g)
    cg_eids = pd.DataFrame(cg.edges(data="edge_id"))[[2]].set_index(2)
    print(len(cg_eids))
    c_edges = edges_all.loc[cg_eids.index.unique()][["osm_id", "edge_id", "geometry", "from_node", "to_node", f"index_{mode}_ft", 
                                                     f"index_{mode}_tf", f"access_{access}_ft", f"access_{access}_tf"]].reset_index()
    c_edges.drop(columns=2, inplace=True)
    c_edges.set_index("edge_id", inplace=True)
    c_nodes = nodes_all[nodes_all.index.isin(n_acc_g)]
    # reset ea_path and init index for adding simplified edges
    ea_path.clear()
    ea_ids["cur"] = c_edges.index.max() + 1 # increment for newly added edge ids
    ea_ids["start"] = c_edges.index.max() + 1
    # return data
    return c_edges, c_nodes, cg.copy() # return subgraph as a copy for parallelisation to work

def _remove_ends(g:nx.MultiDiGraph, nw_df:gpd.GeoDataFrame, nw_col:str, e_rem=None, n_rem=None):
    print("--- starting removal of self-loops and ends ---")
    g:nx.MultiDiGraph = g
    # stats
    print("nodes", g.number_of_nodes())
    print("edges", g.number_of_edges())
    print("self-loops", nx.number_of_selfloops(g))
    print("isolates", nx.number_of_isolates(g))
    # first: remove self-loop edges
    for n in nx.nodes_with_selfloops(g):
        dta = g.get_edge_data(n, n)
        eid = list(dta.keys())[0]
        rmlen = dta[eid]["length"]
        nw_df.loc[n, "len_removed"] += rmlen
        if e_rem is not None:
            e_rem.append(eid)
        g.remove_edge(n, n)
    # create undirected graph (for determining node degree)
    g_undir = g.to_undirected()
    ndeg = pd.DataFrame(g_undir.degree, columns=["nid", "degree"])
    ndeg.set_index("nid", inplace=True)
    n_remove = ndeg[ndeg.degree==1].index.to_list()
    # TODO: filter here based on len_removed or node weight
    print("will remove", len(n_remove), "nodes and adjacent edges")
    for rn in n_remove:
        nn = [n for n in g_undir.neighbors(rn)][0]
        nw_df.loc[nn, nw_col] += nw_df.loc[rn, nw_col]
        # retrieve edge length
        u, v = list(g.edges(rn))[0]
        dta = g.get_edge_data(u, v)
        eid = list(dta.keys())[0]
        rmlen = dta[eid]["length"]
        # track length of removed edges
        nw_df.loc[nn, "len_removed"] += nw_df.loc[rn, "len_removed"] + rmlen
        g.remove_node(rn)
        if n_rem is not None:
            n_rem.append(rn)
        if e_rem is not None:
            e_rem.append(eid)
    nw_df.loc[n_remove, nw_col] = 0
    nw_df.loc[n_remove, "len_removed"] = 0
    print("nodes", g.number_of_nodes())
    print("edges", g.number_of_edges())
    print("self-loops", nx.number_of_selfloops(g))
    print("isolates", nx.number_of_isolates(g))

def net_simplify_remove_ends(g:nx.MultiDiGraph, nodes_weight:gpd.GeoDataFrame, nodes_weight_col:str, es_removed:list=None, ns_removed:list=None, depth:int=5):
    for i in range(depth):
        _remove_ends(g, nodes_weight, nodes_weight_col, es_removed, ns_removed)

def _extract_orig_edges(e, cval):
    # look up e in new edge dict
    collected = {}
    for pvals in ea_path[e]:
        _e = pvals["eid"]
        _u = pvals["fn"]
        _v = pvals["tn"]
        if _e < ea_ids["start"]:
            collected = collected | {(_u, _v, _e): cval}
        else:
            collected = collected | _extract_orig_edges(_e, cval)
    return collected

def centr_add_orig_edges(centr):
    print("adding original edge to centrality results...")
    to_remove = []
    new_centr = centr.copy()
    tmp_e_start_id = ea_ids["start"]
    print("collecting orig. edges...")
    for u, v, e in new_centr:
        if e < tmp_e_start_id:
            # This is an original graph edge. Nothing to do.
            continue
        # This is a temp. introduced edge (net simplification). Collect original edges.
        new_centr = new_centr | _extract_orig_edges(e, new_centr[(u,v,e)])
        to_remove.append((u, v, e))
    print("removal of tmp edges...")
    for tpl in to_remove:
        del new_centr[tpl]
    print("done.")
    return new_centr

def _collect_eattrs(g, es, cost_col):
    path = []
    tlen = 0
    cost = []
    for u, v in es:
        dta = g.get_edge_data(u, v)
        eid = list(dta.keys())[0]
        path.append({"eid":eid, "fn":u, "tn":v})
        tlen += dta[eid]["length"]
        cost.append(dta[eid][cost_col])
    return path, tlen, cost

def net_simplify_merge_segments(g:nx.MultiDiGraph, nodes_weight:gpd.GeoDataFrame, nodes_weight_col:str, cost_col:str, 
                                always_merge_zero_w_nodes=True, seg_merge_threshold=250):
    # get node candidates for removal (and adjacent segment merge)
    #pd.DataFrame(g.degree, columns=["nid", "deg"]).set_index("nid")
    idx, vals = zip(*g.degree)
    s_deg = pd.Series(vals, idx)
    idx, vals = zip(*g.in_degree)
    s_deg_in = pd.Series(vals, idx)
    idx, vals = zip(*g.out_degree)
    s_deg_out = pd.Series(vals, idx)
    deg_df = pd.DataFrame({"deg":s_deg, "deg_in":s_deg_in, "deg_out":s_deg_out, "nw":nodes_weight[nodes_weight_col]})
    # get candidates, ordered by node weight (ascending)
    cands = deg_df[(deg_df.deg==4) & (deg_df.deg_in==deg_df.deg_out)]["nw"].sort_values()
    print(len(cands), "node candidates for segment join")
    for c in cands.index:
        nns = [n for n in g.neighbors(c)]
        if len(nns)==1:
            # TODO: handle case len(nns)==1: loop -> remove c (same approach as for dead-end) - only handle length differently (sum of segment lengths/2 (in and out edges))
            continue
        elif len(nns)!=2:
            raise Exception(f"unexpected number of neighbors for node {c}: {len(nns)}") # TODO: remove if never fired
        a = nns[0]
        b = nns[1]
        es_ft = [(a,c), (c,b)]
        if not always_merge_zero_w_nodes or cands.loc[c]>0:
            #check len of adj edges
            l = 0
            for u, v in es_ft:       
                dta = g.get_edge_data(u, v)
                eid = list(dta.keys())[0]
                l += dta[eid]["length"]
            if l > seg_merge_threshold:
                # skip
                continue
        # merge adj edges
        # ft direction (a -> b)
        path, tl, cost_ft = _collect_eattrs(g, es_ft, cost_col)
        tc = sum(cost_ft)
        _id = ea_ids["cur"]
        g.add_edge(a, b, _id, edge_id=_id, length=tl, 
                    osm_id=None, geometry=None, access_bicycle_ft=None, access_pedestrian_ft=None,
                    index_bike_incwalk_ft=None, inv=None)
        g.edges[a, b, _id][cost_col]=tc
        ea_path[_id] = path
        ea_ids["cur"] += 1
        # tf direction (b -> a)
        path, tl, cost_tf = _collect_eattrs(g, [(b,c), (c,a)], cost_col)
        tc = sum(cost_tf)
        _id = ea_ids["cur"]
        g.add_edge(b, a, _id, edge_id=_id, length=tl, 
                    osm_id=None, geometry=None, access_bicycle_ft=None, access_pedestrian_ft=None,
                    index_bike_incwalk_ft=None, inv=None)
        g.edges[b, a, _id][cost_col]=tc
        ea_path[_id] = path
        ea_ids["cur"] += 1
        # remove node c from graph (and adj. edges)
        g.remove_node(c)
        # assign node weight to adjacent nodes
        nw = nodes_weight.loc[c, nodes_weight_col]
        avg_c_a = (cost_ft[0] + cost_tf[1]) / 2
        avg_c_b = (cost_ft[1] + cost_tf[0]) / 2
        t_avg_c = avg_c_a + avg_c_b
        nodes_weight.loc[a, nodes_weight_col] += nw*(avg_c_b/t_avg_c) #nw/2
        nodes_weight.loc[b, nodes_weight_col] += nw*(avg_c_a/t_avg_c) #nw/2
        nodes_weight.loc[c, nodes_weight_col] = 0 # reset node weight for removed node

def net_simplify(g:nx.MultiDiGraph, nodes_weight:gpd.GeoDataFrame, nodes_weight_col:str, cost_col:str,
                 es_removed:list=None, ns_removed:list=None, remove_ends=True, merge_segments=True,
                 seg_always_merge_zero_w_nodes:bool=True, seg_merge_threshold:int=250):
    # prepare nodes df
    #nodes_weight = nodes_weight.assign(len_removed=0.0)
    if not "len_removed" in nodes_weight.columns:
        nodes_weight["len_removed"]=0.0
    # prepare reporting
    tnw_orig = nodes_weight[nodes_weight_col].sum()
    n_nodes_orig = g.number_of_nodes()
    n_edges_orig = g.number_of_edges()
    
    if remove_ends:
        net_simplify_remove_ends(g, nodes_weight, nodes_weight_col, es_removed, ns_removed)
    if merge_segments:
        net_simplify_merge_segments(g, nodes_weight, nodes_weight_col, cost_col, seg_always_merge_zero_w_nodes, seg_merge_threshold)
    
    # reporting
    n_nodes_s = g.number_of_nodes()
    n_edges_s = g.number_of_edges()
    print(f"Removed {n_nodes_orig-n_nodes_s} (of {n_nodes_orig}) nodes in total ({(n_nodes_orig-n_nodes_s)/n_nodes_orig:.1%})")
    print(f"Removed {n_edges_orig-n_edges_s} (of {n_edges_orig}) edges in total ({(n_edges_orig-n_edges_s)/n_edges_orig:.1%})")
    tnw_a = nodes_weight[nodes_weight_col].sum()
    print(f"Sum of node weights now: {tnw_a:.4f} (orig: {tnw_orig:.4f}). Difference: {(tnw_a - tnw_orig):.4f} (relative: {(tnw_a - tnw_orig)/tnw_orig:.2%})")

def nodes_weight_grid_sample(nodes_weight, node_weight_col, g:nx.Graph, grid_size):
    print(f"sampling node weights using regular grid (d={grid_size})")
    nw = nodes_weight.loc[list(g.nodes)]
    # sub-sample nodes based on regular grid
    xmin = nw.geometry.x.min()
    ymin = nw.geometry.y.min()
    nw["grid_x"] = ((nw.geometry.x - xmin) / grid_size).astype(int)
    nw["grid_y"] = ((nw.geometry.y - ymin) / grid_size).astype(int)
    nw["grid_id"] = nw.grid_y * nw.grid_x.max() + nw.grid_x
    dx = nw.geometry.x - (xmin + nw.grid_x * grid_size + grid_size/2)
    dy = nw.geometry.y - (ymin + nw.grid_y * grid_size + grid_size/2)
    nw["grid_cdist"] = np.sqrt(dx**2 + dy**2)
    #nodes_weight[f"cell_{node_weight_col}"] = nodes_weight.groupby("grid_id")[node_weight_col].sum()
    nw[f"cell_{node_weight_col}"] = nw.groupby("grid_id")[node_weight_col].transform('sum')
    grid_nodes = nw.groupby("grid_id")["grid_cdist"].idxmin()
    nw.loc[grid_nodes, node_weight_col] = nw[f"cell_{node_weight_col}"]
    nw.loc[~nw.index.isin(grid_nodes), node_weight_col] = 0
    print(f"grid nodes: {len(nw.grid_id.unique())}")
    return nw

#### Additional Helpers ####

def poly_from_bounds_list(arr):
    return poly_from_bounds(arr[0], arr[1], arr[2], arr[3])

def poly_from_bounds(left, bottom, right, top):
    return sly.Polygon([(left, bottom), (left, top), (right, top), (right, bottom), (left, bottom)])

def get_aoi(gdf:gpd.GeoDataFrame, buffer=0, use_bbox=False):
    if use_bbox:
        return poly_from_bounds_list(gdf.total_bounds).buffer(buffer)
    return gdf.unary_union.convex_hull.buffer(buffer)

def transform_geom(geom, from_crs, target_srid=4326, inv_xy=False):
    # determine bounding box of network - reproject to 4326 for Overpass query
    ptrans = pyproj.Transformer.from_crs(from_crs, pyproj.CRS(target_srid), always_xy=inv_xy).transform
    return sly.ops.transform(ptrans, geom)

def get_osm_data_qfile(query_file, bbox, file, replace_existing):
    with open(query_file, 'r', encoding="utf-8") as f:
        query_str = f.read()
    get_osm_data(query_str, bbox, file, replace_existing)

def osm_query(query, bbox, file, replace_existing):
    if os.path.isfile(file):
        if replace_existing:
            os.remove(file)
        else:
            print("OSM query: Output file already exists. Skipping download, re-using local file.")
            return
    if isinstance(bbox, (list, dict, tuple)):
        bbox = str(bbox)
    q_str = query.replace("({{bbox}})", bbox)
    curEndpointIndex = 0
    success = False
    data = urllib.parse.urlencode({'data':q_str})
    data = data.encode('ascii')
    while curEndpointIndex < len(overpass_api_endpoints) and not success:
        try:
            #file_name, headers = urllib.request.urlretrieve(overpass_api_endpoints[curEndpointIndex] + "?data=" + urllib.parse.quote_plus(q_str), file)
            file_name, headers = urllib.request.urlretrieve(overpass_api_endpoints[curEndpointIndex], file, data=data)
        except HTTPError as e:
            print(overpass_api_endpoints[curEndpointIndex] + "?data=" + urllib.parse.quote_plus(q_str))
            print(f"HTTPError while trying to download OSM data from '{overpass_api_endpoints[curEndpointIndex]}': Error code {e.code}\n{e.args}\n{e.info()} --> trying again with next available API endpoint...")
            curEndpointIndex+=1
        except KeyboardInterrupt:
            raise Exception(f"OSM download from '{overpass_api_endpoints[curEndpointIndex]}' interrupted by user.")
        except BaseException as e:
            print(f"An unexpected ERROR occured during OSM data download from '{overpass_api_endpoints[curEndpointIndex]}': {e.args}")
            curEndpointIndex+=1
        else:
            success = True
            print(f"Response headers from API call to {overpass_api_endpoints[curEndpointIndex]}: {headers}")
            print(f"OSM Download from {overpass_api_endpoints[curEndpointIndex]} succeeded.")
    if not success:
        raise Exception("OSM data download was not successful. Terminating.")

def get_osm_data(query, bbox, file, replace_existing):
    file_xml = file.replace(".gpkg", ".xml")
    if os.path.isfile(file):
        if replace_existing:
            os.remove(file)
        else:
            print("Polygon download: Geopackage file already exists. Skipping download, re-using local file.")
            return
    osm_query(query, bbox, file_xml, replace_existing)
    # convert to GeoPackage
    #result = subprocess.run(f"ogr2ogr -f 'gpkg' \"{file}\" \"{file_xml}\" ", shell=True, check=True)
    result = subprocess.run(f"ogr2ogr -f gpkg \"{file}\" \"{file_xml}\" ", check=True, capture_output=True, shell=True)
    print(f"ogr2ogr returned code: {result.returncode}")
    
def print_spatial_file_info(file):
    # INFO on available layers and features
    polys = ogr.Open(file)
    for lid in range(0, polys.GetLayerCount()):
        l:ogr.Layer = polys.GetLayerByIndex(lid)
        c = l.GetFeatureCount()
        if c > 0:
            print(f"layer '{l.GetName()}' \n{c} features")
            print(l.GetExtent(), "\n")
   
    
def tessellate(edges, limit=None, id_col="edge_id", out_file=None, segment_dist=0.5):
    if id_col not in edges.columns:
        edges[id_col] = edges.index
    tess = mp.Tessellation(edges, unique_id = id_col, limit=limit, segment=segment_dist)
    t = tess.tessellation
    print(f"Tessellation created {len(tess.multipolygons)} multipolygons.")
    t['full_area'] = t.geometry.area
    
    print("polygon areas:")
    print(t.area.describe())
    t.set_index(id_col, inplace=True, drop=False)
    t.index.name = "index"
    if out_file:
        t.to_file(out_file)
    return t

def add_population(polygon_gdf, dir_pop_raster, stat="sum", out_col="population", out_file=None):
    # zonal statistics to add population data to polygons
    print(f"polygon input CRS: {polygon_gdf.crs}")
    _p_bounds = get_aoi(polygon_gdf, use_bbox=True)
    f_pop_raster = []
    
    for file in Path(dir_pop_raster).glob('**/*.tif'):
        print("Checking file:", file)
        with rio.open(file) as raster:
            # get crs
            # determine extent
            print(raster.crs)
            print(raster.bounds)
            # reproject tessellation extent to raster crs
            p_bounds = transform_geom(_p_bounds, polygon_gdf.crs, raster.crs)
            r_bounds = poly_from_bounds_list(raster.bounds)
            # check within (or overlap)
            # if within, stop search and use raster as single file for zonal stats
            # if overlaps, add to candidate list
            if sly.within(p_bounds, r_bounds):
                print(f"Polygons are fully contained within extent of {file}")
                f_pop_raster.append(file)
                break
            elif sly.intersects(p_bounds, r_bounds):
                print(f"Bounds of polygons intersect with {file}")
                f_pop_raster.append(file)
            else:
                print(f"no intersection with extent of {file}")

    if len(f_pop_raster) < 1:
        raise Exception("No overlap with given population rasters found. Please provide a raster file that covers your chosen AOI.")
    elif len(f_pop_raster) > 1:
        raise Exception("NotImplemented: your AOI overlaps with multiple raster files. This is not yet supported. Please merge rasters beforehand.")

    # prepare polygon GDF
    gdf_index = polygon_gdf.index
    polygon_gdf.reset_index(inplace=True, drop=True)
    
    for rfile in f_pop_raster:
        with rio.open(rfile) as raster:
            # reproject polygons to raster crs
            p_repr = polygon_gdf.to_crs(raster.crs)
            pop_out = rasterstats.gen_zonal_stats(p_repr, rfile, stats=[stat])#, geojson_out=True)
        # TODO: implement handling of multiple runs of zonal stats (several raster files)
    
    print("--- collecting output ---")
    pop_out = pd.DataFrame.from_records(pop_out)
    print(pop_out["sum"].describe())
    if len(polygon_gdf) != len(pop_out):
        print(f"length mismatch of input and output. polygons: {len(polygon_gdf)}, output: {len(pop_out)}")
    if out_col in polygon_gdf.columns:
        polygon_gdf.drop(columns=[out_col], inplace=True)
        print(f"Overwriting existing column '{out_col}'")
    result = polygon_gdf.join(pop_out.rename(columns={stat:out_col}))
    result.set_index(gdf_index, inplace=True)
    if out_file:
        result.to_file(out_file)
    return result

def _compute_node_weights(edges, edge_weights, weight_col = "area", output_col="weight"):
    # join network (edge list) f/t node columns with edge weight column
    tmp = edges[['from_node','to_node', 'geometry']].join(edge_weights[[weight_col]]) # edge-based weight
    #display(tmp.explore(column="area"))
    tmp.drop(columns='geometry', inplace=True)
    wa = tmp.groupby(['from_node']).sum()[[weight_col]]/2.0 # split edge weight to end nodes per edge
    wa.index.rename("nid", inplace=True)
    wb = tmp.groupby(['to_node']).sum()[weight_col]/2.0
    wb.index.rename("nid", inplace=True)
    w = wa.join(wb, lsuffix='_a', rsuffix='_b', how='outer') # begin aggregation for each node role (from/to)
    w.fillna(0.0, inplace=True) # set zero weight for nodes with non-existing weight input data
    w[output_col] = w[f"{weight_col}_a"] + w[f"{weight_col}_b"]
    dif = w[output_col].sum() - edge_weights[[weight_col]].sum().iloc[0]
    print("Sum of node weights", "equals" if abs(dif) < 0.0001 else "DOES NOT MATCH", "sum of edge weights")
    print(f">>> total dif: {dif:.4f} --- sum of node weights: {w[output_col].sum():.2f}, sum of edge weights: {edge_weights[[weight_col]].sum().iloc[0]:.2f}")
    print(f">>> this equals to {dif/edge_weights[[weight_col]].sum().iloc[0]:.1%}")
    return w

def add_node_weights(nodes: gpd.GeoDataFrame, edges: gpd.GeoDataFrame, edge_weights, weight_col = "area", output_col="weight", output_file=None):
    # remove output col if already exists
    if output_col in nodes.columns:
        nodes.drop(columns=[output_col, f"{weight_col}_a", f"{weight_col}_b"])
    result = nodes.join(_compute_node_weights(edges, edge_weights, weight_col, output_col))
    if output_file:
        result.to_file(output_file)
    return result

### Priority Score helper functions

def p1(b, c, t=1): ## std (with cutoff)
    return (t - b) / t * c

def p2(b, c, t=1): ## v2c
    return (1 - np.sqrt(1 / t * b)) * c

def p3(b, c, t=1):
    return ((t - b) / t) ** 2 * c

ps_functions = {
    "p1": p1,
    "p2": p2,
    "p3": p3
}

def ps(net, mode, centr, f, ps_name, t=1):
    cn = f"ps_{mode}_{centr}_{ps_name}"
    if t < 1:
        cn += f"_t{t}"
    cn_ft = f"{cn}_ft"
    cn_tf = f"{cn}_tf"
    c_col_ft = f"centr_{centr}_norm_ft"
    c_col_tf = f"centr_{centr}_norm_tf"
    b_col_ft = f"index_{mode}_ft"
    b_col_tf = f"index_{mode}_tf"
    net.loc[:,cn_ft] = f(net[b_col_ft], net[c_col_ft], t)
    net.loc[:,cn_tf] = f(net[b_col_tf], net[c_col_tf], t)
    # if t < 1: set ps to zero for rows with b > t
    if t < 1:
        net.loc[net[b_col_ft] > t, cn_ft] = 0
        net.loc[net[b_col_tf] > t, cn_tf] = 0
    # append mean value (dir-indep.)
    net.loc[:,cn] = net[[cn_ft, cn_tf]].mean(1)    
    
def compute_priorities(net, mode, centr, t=1):
    # normalize centrality
    max_bc = net[f"centr_{centr}_ft"].max()
    tmp = net[f"centr_{centr}_tf"].max()
    if tmp > max_bc:
        max_bc = tmp
    net.loc[:,f"centr_{centr}_norm_ft"] = net[f"centr_{centr}_ft"] / max_bc
    net.loc[:,f"centr_{centr}_norm_tf"] = net[f"centr_{centr}_tf"] / max_bc
    # compute priority scores
    for k in ps_functions:
        ps(net, mode, centr, ps_functions[k], k, t)