# INFO: this code is based on the NetworkX centrality implementation: https://github.com/networkx/networkx.
# It is an improved, advanced version of the spatial betweenness centrality implementation by 
# Werner and Loidl 2023: https://doi.org/10.5281/zenodo.8125632, which was derived from the 
# SIBC implementation by Wu et al. 2022: https://doi.org/10.6084/m9.figshare.19402562

"""Advanced Betweenness centrality measures."""
from heapq import heappop, heappush
from itertools import count

import networkx as nx
from networkx.algorithms.shortest_paths.weighted import _weight_function
from networkx.utils.decorators import not_implemented_for

import pandas as pd
import geopandas as gpd
import numpy as np

import multiprocessing
from timeit import default_timer as timer

# linear, single-process function
def _spatial_betweenness_centrality(G, nodes_gdf, weight_col, from_nodes, 
                                                    w_orig, w_dest, normalized, od_w_norm_ref,
                                                    cost_cutoff, dist_cutoff, net_dist_filter, node_weight_col, dist_decay_from,
                                                    random_sample_n=None):
    print("computing weighted betweenness centrality for", len(from_nodes), "source nodes and up to", len(G), "destination nodes (single process)...")
    if dist_decay_from:
        raise Exception("dist_decay_from is not implemented.")
    if random_sample_n is not None and random_sample_n > 0:
        if random_sample_n > len(G.nodes):
            print("random sample n is greater than number of nodes. Using full graph instead")
        else:
            print("using random subsample of", random_sample_n, "nodes.")
            from_nodes = np.random.choice(from_nodes, size=random_sample_n, replace=False)
    
    tstart = timer()
    # progress reporting
    _ncount = len(from_nodes)
    _progr_i = int(len(from_nodes) / 100)
    if _progr_i > 100:
        _progr_i = 100
    elif _progr_i > 1000:
        _progr_i = 10
    _progr_s = False
    _i = 0
    if _progr_i > 0:
        _progr_s = True
    # init
    betweenness = dict.fromkeys(G.edges(), 0.0)
    # iterate over source (origin) nodes
    for s in from_nodes:
        if _progr_s and (_i%_progr_i==0):
            tdif = (timer() - tstart)/60
            share = _i/_ncount
            if share > 0:
                print(f"{_i}/{_ncount}      {share:.1%}      elapsed: {tdif:.1f} min (est: {(1/share-1)*tdif:.1f})", end='\r')
        _i+=1
        # apply spatial filter to reduce network size
        td_ids = None
        if net_dist_filter:
            if dist_cutoff and net_dist_filter < dist_cutoff:
                raise Exception("you specified a smaller net distance filter than distance cutoff")
            td = nodes_gdf.geometry.distance(nodes_gdf.loc[s].geometry)
            td_ids = td[td <= net_dist_filter].index
            # OPTION: filter for connected component
        # single source shortest paths # use Dijkstra's algorithm 
        # OPTION: here, distance-restricted Dijkstra could be used instead -> e.g. NetworkX single_source_dijkstra
        # however, the default NetworkX implementation does not return sigma - requires changes
        # for now: using euclidean straigt-line distance threshold to pre-filter graph
        S, P, sigma, cost, dist = _single_source_dijkstra_path_basic_s(G if td_ids is None else G.subgraph(td_ids), s, weight_col)
        # additional network (cost-) distance limit: exclude dest nodes that exceed the network (cost-) distance limit from accumulation
        skip_dest_nodes = []
        if cost_cutoff:
            _skip_dest_nodes = pd.DataFrame.from_dict(cost, 'index', columns=["ccost"])
            skip_dest_nodes = _skip_dest_nodes[_skip_dest_nodes.ccost > cost_cutoff].index.to_list()
        if dist_cutoff:
            _skip_dest_nodes = pd.DataFrame.from_dict(dist, 'index', columns=["cdist"])
            skip_dest_nodes += _skip_dest_nodes[_skip_dest_nodes.cdist > dist_cutoff].index.to_list()
        # define interaction (weight per o-d relation)
        d_weights = _compute_od_weights(w_orig, w_dest, s, od_w_norm_ref, node_weight_col)
        # accumulation
        betweenness = _accumulate_edges_spatial_cent(betweenness, S, P, sigma, d_weights, skip_dest_nodes)
    
    print(f"{_i}/{_ncount}      {_i/_ncount:.1%} - done with routing.                         ")

    # rescale/normalize: by default (spatial betweenness centr.): divides betweenness value by sum of origin weights
    # if normalize = False, then betweenness is returned unscaled
    o_w_sum = od_w_norm_ref if od_w_norm_ref else w_orig[[node_weight_col]].sum()[node_weight_col]
    print("\nrescaling...")
    betweenness = _rescale_e_spatial_cent(
        betweenness, o_w_sum, len(G), normalized=normalized, directed=G.is_directed()
    )
    print("adding edge keys...")
    if G.is_multigraph():
        betweenness = _add_edge_keys(G, betweenness, weight=weight_col)
    tend = timer()
    print(f"--- Centrality computation TOTAL TIME: {(tend-tstart)/60:.2f} min ---")
    return betweenness


# parallel computing version

class _BC_Internal_:
    def __init__(self, G, nodes_gdf, weight_col, node_weight_col, net_dist_filter, dist_cutoff, cost_cutoff, w_orig, w_dest, od_w_norm_ref, decay_from):
        self.G = G
        self.nodes_gdf = nodes_gdf
        self.weight_col = weight_col
        self.node_weight_col = node_weight_col
        self.net_dist_filter = net_dist_filter
        self.dist_cutoff = dist_cutoff
        self.cost_cutoff = cost_cutoff
        self.w_orig = w_orig
        self.w_dest = w_dest
        self.od_w_norm_ref = od_w_norm_ref
        self.decay_from = decay_from

    def process_source_node(self, s):
        # apply spatial filter to reduce network size
        td_ids = None
        td = None
        w_dest_override = None
        if self.net_dist_filter:
            if self.dist_cutoff and self.net_dist_filter < self.dist_cutoff:
                raise Exception("you specified a smaller net distance filter than distance cutoff")
            td = self.nodes_gdf.geometry.distance(self.nodes_gdf.loc[s].geometry)
            td_ids = td[td <= self.net_dist_filter].index
            # OPTION: filter for connected component
        # optional: weight-decay based on distance (similar to gravity function)            
        # single source shortest paths # use Dijkstra's algorithm 
        # OPTION: here, distance-restricted Dijkstra could be used instead -> e.g. NetworkX single_source_dijkstra
        # however, the default NetworkX implementation does not return sigma - requires changes
        # for now: using euclidean straigt-line distance threshold to pre-filter graph
        g = self.G if td_ids is None else self.G.subgraph(td_ids)
        S, P, sigma, cost, dist = _single_source_dijkstra_path_basic_s(g, s, self.weight_col)
        # additional network (cost-) distance limit: exclude dest nodes that exceed the network (cost-) distance limit from accumulation
        skip_dest_nodes = []
        if self.cost_cutoff:
            _skip_dest_nodes = pd.DataFrame.from_dict(cost, 'index', columns=["ccost"])
            skip_dest_nodes = _skip_dest_nodes[_skip_dest_nodes.ccost > self.cost_cutoff].index.to_list()
        if self.dist_cutoff:
            _nodes_dist = pd.DataFrame.from_dict(dist, 'index', columns=["cdist"])
            skip_dest_nodes += _nodes_dist[_nodes_dist.cdist > self.dist_cutoff].index.to_list()
            # distance decay (o-d weights)
            if self.decay_from:
                #if td is None:
                #    td = self.nodes_gdf.geometry.distance(self.nodes_gdf.loc[s].geometry)
                w_dest_override = self.w_dest.loc[_nodes_dist[_nodes_dist.cdist<=self.dist_cutoff].index, self.node_weight_col].copy()
                _decay_d = self.dist_cutoff - self.decay_from
                w_dest_override.loc[(_nodes_dist[(_nodes_dist.cdist>self.decay_from) & (_nodes_dist.cdist<=self.dist_cutoff)]).index] *= (
                    (self.dist_cutoff - _nodes_dist.loc[(_nodes_dist.cdist>self.decay_from).index].cdist) / _decay_d
                    )
                #tmp = w_dest_override.loc[~w_dest_override.index.isin(skip_dest_nodes)]
                # if len(tmp[tmp<0] > 0):
                #    print(tmp[tmp<0])
                #    import matplotlib.pyplot as plt
                #    plt.scatter(x=_nodes_dist.loc[tmp.index].cdist, y=tmp)
                #    plt.show()
                #    raise Exception("found weight < 0")
        #define interaction (weight per o-d relation)
        d_weights = _compute_od_weights(self.w_orig, self.w_dest if w_dest_override is None else w_dest_override, s, self.od_w_norm_ref, self.node_weight_col)
        return _accumulate_edges_spatial_cent(dict.fromkeys(g.edges(), 0.0), S, P, sigma, d_weights, skip_dest_nodes) 


def spatial_betweenness_centrality(g, nodes_gdf, weight_col, from_nodes=None, 
                                        w_orig=None, w_dest=None, normalized=True, od_w_norm_ref=None,
                                        cost_cutoff=None, dist_cutoff=None, net_dist_filter=None, node_weight_col="weight", 
                                        processes=None, chunksize=10, tasks_per_child=None, dist_decay_from=None, random_sample_n=None):
    if not weight_col:
        raise Exception("This method is only defined for weighted routing -> please specify a 'weight' column name")
    if w_orig is None and node_weight_col in nodes_gdf.columns:
        w_orig = nodes_gdf
    elif w_orig is None:
        raise Exception("Please provide origin weights df (w_orig) or a 'weight' column in nodes_gdf.")
    if w_dest is None and node_weight_col in nodes_gdf.columns:
        w_dest = nodes_gdf
    elif w_dest is None:
        raise Exception("Please provide destination weights df (w_dest) or a 'weight' column in nodes_gdf.")
    if net_dist_filter is None and dist_cutoff:
        net_dist_filter = dist_cutoff
    # if source nodes are not defined, compute centrality using all nodes as sources
    if from_nodes is None:
        from_nodes = nodes_gdf[w_orig[node_weight_col]>0].index
    print(f"""Betweenness centrality function uses the following parameters:
        routing weight column: {weight_col}
        node weight column:    {node_weight_col}
        normalized:            {normalized}
        od_w_norm_ref:         {od_w_norm_ref}
        cost_cutoff:           {cost_cutoff}
        dist_cutoff:           {dist_cutoff}
        net_dist_filter:       {net_dist_filter}
        dist_decay:            {dist_decay_from}
        """)
    # run single-process version if requested
    if processes == 1:
        return _spatial_betweenness_centrality(g, nodes_gdf, weight_col, from_nodes, w_orig, w_dest, normalized, od_w_norm_ref, 
                                                    cost_cutoff, dist_cutoff, net_dist_filter, node_weight_col, dist_decay_from, random_sample_n=random_sample_n)
    # run parallelized version
    print("computing weighted betweenness centrality for", len(from_nodes), "source nodes and up to", len(g), "destination nodes (parallelized)...")
    tstart = timer()

    # random sampling (use with caution - only for testing or std ebc approximation)
    if random_sample_n is not None and random_sample_n > 0:
        if random_sample_n > len(g.nodes):
            print("random sample n is greater than number of nodes. Using full graph instead")
        else:
            print("using random subsample of", random_sample_n, "nodes.")
            from_nodes = np.random.choice(from_nodes, size=random_sample_n, replace=False)
            
    # progress reporting
    _ncount = len(from_nodes)
    _i = 0
            
    # init
    betweenness = dict.fromkeys(g.edges(), 0.0) # only init for edges, not for nodes

    # parallel processing: iterate over source (origin) nodes
    fn = _BC_Internal_(g, nodes_gdf, weight_col, node_weight_col, net_dist_filter, dist_cutoff, cost_cutoff, w_orig, w_dest, od_w_norm_ref, dist_decay_from).process_source_node
    pool = multiprocessing.Pool(processes=processes, maxtasksperchild=tasks_per_child)
    results = pool.imap(fn, from_nodes, chunksize=chunksize)
    
    # collect and combine results
    for betw in results:
        for e in betw:
            betweenness[e] += betw[e]
        _i += 1
        if _i%10 < 1:
            tdif = (timer() - tstart)/60
            share = _i/_ncount
            if share > 0:
                print(f"  {_i}/{_ncount}    {share:.1%}       elapsed: {tdif:.1f} min (est: {(1/share-1)*tdif:.1f})  ", end='\r')
    pool.close()
    pool.join()

    print(f"  {_i}/{_ncount}    {_i/_ncount:.1%} - done with routing.                              ")

    # rescale/normalize: by default (spatial betweenness centr.): divides betweenness value by sum of origin weights
    # if normalize == False, then betweenness is returned unscaled
    o_w_sum = od_w_norm_ref if od_w_norm_ref else w_orig[[node_weight_col]].sum()[node_weight_col]
    print("\nrescaling...")
    betweenness = _rescale_e_spatial_cent(
        betweenness, o_w_sum, len(g), normalized=normalized, directed=g.is_directed()
    )
    print("adding edge keys...")
    if g.is_multigraph():
        betweenness = _add_edge_keys(g, betweenness, weight=weight_col)
    tend = timer()
    print(f"--- Centrality computation TOTAL TIME: {(tend-tstart)/60:.2f} min ---")
    return betweenness


# helpers for betweenness centrality

# compute weights per o-d relation
def _compute_od_weights(orig_weights, dest_weights, orig_node, max_destw=None, node_weight_col="weight"):
    if type(dest_weights) is gpd.GeoDataFrame:
        _d_w = dest_weights[node_weight_col]
    elif type(dest_weights) is pd.Series:
        _d_w = dest_weights
    else:
        raise Exception(f"Unsupported type '{type(dest_weights)}' provided. Implemented for GeoDataFrame and Series only.")
    orig_w = orig_weights[node_weight_col][orig_node]
    if max_destw:
        sum_dw = max_destw - _d_w[orig_node]
    else:
        sum_dw = _d_w[_d_w>0].sum() - _d_w[orig_node]
    #dest_weights['d_w'] = dest_weights.weight / sum_dw * orig_w
    #d_w = dest_weights[['d_w']].to_dict()['d_w']
    d_w = (_d_w / sum_dw * orig_w).to_dict() 
    d_w[orig_node] = 0
    return d_w

def _single_source_dijkstra_path_basic_s(G, s, weight):
    weight = _weight_function(G, weight)
    distance = _weight_function(G, "length")
    # modified from Eppstein
    S = []
    P = {}
    for v in G:
        P[v] = []
    sigma = dict.fromkeys(G, 0.0)  # sigma[v]=0 for v in G
    D = {}
    DIST = {}
    sigma[s] = 1.0
    push = heappush
    pop = heappop
    seen = {s: 0}
    c = count()
    Q = []  # use Q as heap with (distance,node id) tuples
    push(Q, (0, 0, next(c), s, s))
    while Q:
        (cost, dist, _, pred, v) = pop(Q)
        if v in D:
            continue  # already searched this node.
        sigma[v] += sigma[pred]  # count paths
        S.append(v)
        D[v] = cost
        DIST[v] = dist
        for w, edgedata in G[v].items():
            vw_cost = cost + weight(v, w, edgedata)
            vw_dist = dist + distance(v, w, edgedata)
            if w not in D and (w not in seen or vw_cost < seen[w]):
                seen[w] = vw_cost
                push(Q, (vw_cost, vw_dist, next(c), v, w))
                sigma[w] = 0.0
                P[w] = [v]
            elif vw_cost == seen[w]:  # handle equal paths
                sigma[w] += sigma[v]
                P[w].append(v)
    # S: vertices; P: path reconstr; sigma: path count; D: distances
    return S, P, sigma, D, DIST

def _accumulate_edges_spatial_cent(betweenness, S, P, sigma, d_weights, skip_dest_nodes):
    delta = dict.fromkeys(S, 0)
    while S:
        # iterate over destination nodes
        # w: dest node
        w = S.pop()
        if w in skip_dest_nodes:
            continue
        # debug
        if d_weights[w] < 0:
            raise Exception(f"negative weight for node {w}")
        #
        coeff = (d_weights[w]+delta[w]) / sigma[w]
        for v in P[w]:
            c = sigma[v] * coeff
            betweenness[(v, w)] += c
            delta[v] += c
    return betweenness

def _rescale_e_spatial_cent(betweenness, total_weight, n, normalized, directed=False, k=None):
    if normalized:
        if n <= 1:
            scale = None  # no normalization b=0 for all nodes
        else:
            scale = 1 / total_weight
    else:  # rescale by 2 for undirected graphs
        if not directed:
            scale = 0.5
        else:
            scale = None
    if scale is not None:
        if k is not None:
            scale = scale * n / k
        for v in betweenness:
            betweenness[v] *= scale
    return betweenness

@not_implemented_for("graph")
def _add_edge_keys(G, betweenness, weight=None):
    r"""Adds the corrected betweenness centrality (BC) values for multigraphs.

    Parameters
    ----------
    G : NetworkX graph.

    betweenness : dictionary
        Dictionary mapping adjacent node tuples to betweenness centrality values.

    weight : string or function
        See `_weight_function` for details. Defaults to `None`.

    Returns
    -------
    edges : dictionary
        The parameter `betweenness` including edges with keys and their
        betweenness centrality values.

    The BC value is divided among edges of equal weight.
    """
    _weight = _weight_function(G, weight)

    edge_bc = dict.fromkeys(G.edges, 0.0)
    for u, v in betweenness:
        d = G[u][v]
        wt = _weight(u, v, d)
        keys = [k for k in d if _weight(u, v, {k: d[k]}) == wt]
        bc = betweenness[(u, v)] / len(keys)
        for k in keys:
            edge_bc[(u, v, k)] = bc

    return edge_bc