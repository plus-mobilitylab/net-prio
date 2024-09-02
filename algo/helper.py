import geopandas as gpd
import pandas as pd
import contextily as cx
import matplotlib.pyplot as plt
import os
import numpy as np
import plotly.express as px

class CentralityDef:
    
    @staticmethod
    def from_str(name):
        _offset_ = 0
        parts = name.split("_")
        c_type = parts[1]
        # detect decay version
        decay_from = -1
        if parts[2] == "dec":
            decay_from = int(parts[3])
            c_cut = parts[4]
            _offset_ = 2
            c_cut = int(parts[2+_offset_])
        else:
            c_cut = int(parts[2+_offset_][1:])
        c_is_bp = parts[3+_offset_] == "bp"
        c_dfac = 0
        c_nws = 0
        refnet = False
        _i = 5 + _offset_
        if c_is_bp:
            _i += 1
            c_dfac = int(parts[4+_offset_][1:])
        if _i+1 < len(parts) and parts[_i] != "refnet":
            c_nws = int(parts[_i])
        if parts[-2] == "refnet":
            refnet = True
        #print("type:", c_type, "cut:", c_cut, "is_bp:", c_is_bp, "dfac:", c_dfac, "nws:", c_nws)
        return CentralityDef(c_type, c_cut, c_is_bp, c_dfac, c_nws, refnet, decay_from)
    
    type:str = None
    cut:int = 0
    decay_from:int = -1
    is_bp:bool = False
    dfac:int = 0
    nws:int = 0
    refnet:bool = False
    
    def __init__(self, type = None, cut = 0, is_bp = False, dfac = 0, nws = 0, refnet=False, decay_from=-1):
        if len(type) > 10:
            print("WARN: your specified 'type' parameter is unexpectedly long -> did you intend to call the 'from_str()' method?")
        self.type = type
        self.cut = cut
        self.is_bp = is_bp
        self.dfac = dfac
        self.nws = nws
        self.refnet = refnet
        self.decay_from = decay_from
    
    def clone(self):
        return CentralityDef.from_str(self.to_str())
    
    def to_str(self):
        if self.decay_from > -1:
            return f'centr_{self.type}_dec_{self.decay_from}_{self.cut}_{"bp_d" if self.is_bp else "sp"}{self.dfac if self.is_bp else ""}{f"_nws_{self.nws}" if self.nws > 0 else ""}{"_refnet" if self.refnet else ""}'
        return f'centr_{self.type}_c{self.cut}_{"bp_d" if self.is_bp else "sp"}{self.dfac if self.is_bp else ""}{f"_nws_{self.nws}" if self.nws > 0 else ""}{"_refnet" if self.refnet else ""}'
    def __str__(self):
        return self.to_str()
    
    @property
    def name_sum(self):
        return f"{self}_sum"
    @property
    def name_ft(self):
        return f"{self}_ft"
    @property
    def name_tf(self):
        return f"{self}_tf"
    
def get_centr_compare_cols(df:gpd.GeoDataFrame, c:CentralityDef, cref:CentralityDef):
    vals_c = df[c.name_sum].fillna(0)
    vals_cref = df[cref.name_sum].fillna(0)
    # change: absolute difference (non-normalized, absolute centrality values)
    d_abs = vals_c - vals_cref
    # change: relative to original centrality value
    d_rel = d_abs / vals_cref
    # change: share of max. centrality
    dn_abs = vals_c / vals_c.max() - vals_cref / vals_cref.max()
    # change: relative to original centr. share
    dn_rel = dn_abs / (vals_cref / vals_cref.max())
    return pd.DataFrame({"d_abs":d_abs, "d_rel":d_rel, "dn_abs":dn_abs, "dn_rel":dn_rel})

def save_plot(name, dir_detail_plot, show=False, dpi=600):
    if not os.path.exists(dir_detail_plot):
        os.makedirs(dir_detail_plot)
    plt.savefig(fname=os.path.join(dir_detail_plot, f"{name}.png"), bbox_inches='tight', dpi=dpi)
    if show:
        plt.show()
    plt.clf()
    plt.close()
    
def get_aoi_extent(aoi_name, aoi_file=None, get_full_extent=False, reproj=True, custom_crs=None) -> gpd.GeoDataFrame:
    if aoi_file is None:
        aoi_file = os.path.join("data", "aois.gpkg")
    if not os.path.exists(aoi_file):
        raise Exception(f"ERROR: AOI geopackage '{aoi_file}' not found")
    layer = "core_extent_wgs84"
    if get_full_extent:
        layer = "full_extent_wgs84"
    aois = gpd.read_file(aoi_file, columns=["aoi_name", "srid", "geom"], layer=layer).set_index("aoi_name")
    if reproj:
        if custom_crs is None:
            custom_crs = aois.loc[aoi_name,"srid"]
        return aois.loc[[aoi_name],:].to_crs(custom_crs)
    return aois.loc[[aoi_name],:]
    

def centr_comparison(aoi_name:str, centr_df:gpd.GeoDataFrame, c:CentralityDef, cref:CentralityDef, 
                     c_quant=0.9, c_ofmax=0.2, dir_detail_plot="plots_detail", centr_diff_name=None, 
                     c_label=None, cref_label=None, ccomp_label=None,
                     aoi_file=None, generate_plots=True, plot_all=False):
    if c.name_sum == cref.name_sum:
        raise Exception("Both centrality colums supplied are identical - please provide differing column names!")       
    
    if c.name_sum not in centr_df.columns:
        print(f"WARNING: centrality column '{c.name_sum}' is not a valid column name of the input df")
        return None
    if cref.name_sum not in centr_df.columns:
        print(f"WARNING: centrality column '{cref.name_sum}' is not a valid column name of the input df")
        return None

    print("AOI", aoi_name, "- comparing", c, "to", cref)
    if ccomp_label:
        print(" >>>", ccomp_label)
    
    if not centr_diff_name:
        _n_ = "cdif"
        if c.type != cref.type:
            _n_ = f"{_n_}_{cref.type}"
        if c.cut != cref.cut:
            _n_ = f"{_n_}_c{cref.cut}"
        if c.decay_from != cref.decay_from:
            _n_ = f"{_n_}_dec_{cref.decay_from}_{cref.cut}"
        if c.is_bp != cref.is_bp:
            _n_ = f"{_n_}_{'bp' if cref.is_bp else 'sp'}"
        elif c.dfac != cref.dfac:
            _n_ = f"{_n_}_d{cref.dfac}"
        if c.nws != cref.nws:
            _n_ = f"{_n_}_nws{cref.nws}"
        if c.refnet != cref.refnet:
            _n_ = f"{_n_}_refnet"
        centr_diff_name = _n_
    
    out_fname_id = c 
    if c_label is not None:
        out_fname_id = c_label
        
    if cref_label is None:
        cref_label = cref.to_str()
    if c_label is None:
        c_label = c.to_str()
    if ccomp_label is None:
        ccomp_label = f"{c}, {cref}"
    
    pd.options.plotting.backend = "matplotlib"
    
    if generate_plots:
        # create subdir centr_diff_name
        dir_detail_plot = os.path.join(dir_detail_plot, centr_diff_name)
        if not os.path.exists(dir_detail_plot):
            os.makedirs(dir_detail_plot)
    
    # show relevant (high-centr) segments (based on reference column)
    c_trh_q = centr_df[cref.name_sum].fillna(0).quantile(c_quant)
    c_trh_p = centr_df[cref.name_sum].max() * c_ofmax
    print(f"centrality stats: max={centr_df[cref.name_sum].max():0.6f}")
    print(f"high centrality cutoff values: \n  quantile({c_quant}):     \t{c_trh_q:0.6f} \n  {c_ofmax:.1%} of max.:\t{c_trh_p:0.6f}")
    
    # add diff columns (c vs. cref)
    d_df = get_centr_compare_cols(centr_df, c, cref)
    
    if generate_plots and plot_all:
        # plot difference values (hist)
        d_df.d_rel.hist(range=[-1,1])
        save_plot(f"{c}__drel_hist", dir_detail_plot)
        d_df.dn_rel.hist(range=[-1,1])
        save_plot(f"{c}__dnrel_hist", dir_detail_plot)
        d_df.dn_abs.hist()
        save_plot(f"{c}__dnabs_hist", dir_detail_plot)
    
    # normalized difference: relative
    print(c)
    temp_vis = centr_df.join(d_df).replace([np.inf, -np.inf], np.nan)
    
    # relative diff stats
    ## segments with high centrality (defined relative to each centr.'s max value)
    hc_seg = centr_df[centr_df[c.name_sum] > centr_df[c.name_sum].max()*c_ofmax]
    hc_seg_ref = centr_df[centr_df[cref.name_sum] > c_trh_p]
    d_max_c = centr_df[c.name_sum].max() - centr_df[cref.name_sum].max()
    hc_seg_n = len(hc_seg)
    hc_seg_ref_n = len(hc_seg_ref)
    hc_seg_d_n = hc_seg_n - hc_seg_ref_n
    print("\nChanges induced by test case:")
    print(f"max. centrality changed by {d_max_c:.6f} ({d_max_c/centr_df[cref.name_sum].max():.1%})")
    print(f"high-centr. segments (centr > {c_ofmax:.1%} of respective max): {hc_seg_ref_n} ({hc_seg_n} for test case). Delta: {hc_seg_d_n} ({hc_seg_d_n/hc_seg_ref_n:.1%})")

    # Test case introduced the following high-centr. segments:
    hc_seg_added = hc_seg[~hc_seg.index.isin(hc_seg_ref.index)]
    print(f"Test case added", len(hc_seg_added), "high-centr. segments")
    #hc_seg_added.explore(tiles="CartoDB Positron", tooltip=[cref.name_sum, c.name_sum])

    # Test case removed the following high-centr. segments:
    hc_seg_removed = hc_seg_ref[~hc_seg_ref.index.isin(hc_seg.index)]
    print(f"Test case removed", len(hc_seg_removed), "high-centr. segments")
    #hc_seg_removed.explore(tiles="CartoDB Positron", tooltip=[cref.name_sum, c.name_sum])
    
    hc_seg_remained = hc_seg_ref[hc_seg_ref.index.isin(hc_seg.index)]
    n_hc_seg_remained = len(hc_seg_remained)
    
    if generate_plots:
        # static plot:
        map_ax = hc_seg.plot(figsize=(10, 10), edgecolor="#bbbbbb")
        map_ax = hc_seg_remained.plot(ax=map_ax, figsize=(10, 10), edgecolor="#999")
        map_ax.set_title(f'High-centr. segments\n({ccomp_label})', fontdict={'fontsize': 15}, loc='center')
        map_ax = hc_seg_added.plot(ax=map_ax, figsize=(10, 10), edgecolor="#225E85")
        map_ax = hc_seg_removed.plot(ax=map_ax, edgecolor="#D02F2B", zorder=0)
        set_plot_aoi(map_ax, aoi_name, True, True, zoom_buffer=-5000, add_basemap=True) #basemap_zoom_adjust=0
        plot_map(map_ax,f"{out_fname_id}__hc_segments", dir_detail_plot)
    
    # add absolute values for dn_abs -> for sorting / plot order
    temp_vis.loc[:,"dn_abs_pos"] = temp_vis.loc[:,"dn_abs"].abs()
     
    # normalized difference: absolute - e.g. 0.2 means that difference equals 20% of maximum centrality value
    orig_hc_q = temp_vis[temp_vis[cref.name_sum] > c_trh_q]
    t_max_abs = orig_hc_q.dn_abs.abs().max()
    
    if generate_plots and plot_all:
        # static plot: dn_abs for HC-segments (upper q)
        map_ax = orig_hc_q.sort_values(by="dn_abs_pos").plot(column="dn_abs", vmin=-t_max_abs, vmax=t_max_abs, cmap="RdYlBu", legend=True, figsize=(10, 10),
                        legend_kwds={"shrink":0.2, "pad":0.01}
                    )
        map_ax.set_title(f'Absolute normalized difference for high-centr. (q={c_quant}) segments\n({ccomp_label})', fontdict={'fontsize': 12}, loc='center')
        set_plot_aoi(map_ax, aoi_name, True, True, zoom_buffer=-5000, add_basemap=True) #basemap_zoom_adjust=0
        plot_map(map_ax, f"{out_fname_id}__hc_dn_abs", dir_detail_plot)
    
    # segments with increase (absolute value) of more than 10% of highest centrality value
    seg_dn_abs_n_gt10 = sum(temp_vis.dn_abs > 0.1)
    print(f"Segments with increase greather than 10% of max. centrality value: {seg_dn_abs_n_gt10}")
    
    if generate_plots:
        if plot_all:
            # static plot of HC increase:
            map_ax = temp_vis[temp_vis.dn_abs > 0.1].plot(edgecolor="#225E85", figsize=(10, 10),
                            legend_kwds={"shrink":0.2, "pad":0.01}
                        )
            map_ax.set_title(f'Segments with high centrality increase (10% of max.)\n({ccomp_label}) n={len(temp_vis[temp_vis.dn_abs > 0.1])} (of {len(centr_df)})', fontdict={'fontsize': 12}, loc='center')
            set_plot_aoi(map_ax, aoi_name, True, True, zoom_buffer=-5000, add_basemap=True) #basemap_zoom_adjust=0
            plot_map(map_ax, f"{out_fname_id}__hc_dn_abs_incr10", dir_detail_plot)
            
        # centrality comparison plot
        fig, axes = plt.subplots(ncols=3, figsize=(30,30))
        axes[0].set_title(cref_label if cref_label is not None else cref, fontdict={'fontsize': 15}, loc='center')
        axes[1].set_title(c_label if c_label is not None else c, fontdict={'fontsize': 15}, loc='center')
        axes[2].set_title("absolute diff: normalized", fontdict={'fontsize': 15}, loc='center')
        axes[0].set_axis_off()
        axes[1].set_axis_off()
        axes[2].set_axis_off()
        temp_vis[temp_vis[cref.name_sum]>0].sort_values(by=cref.name_sum).plot(column=cref.name_sum, ax=axes[0], cmap="Reds")
        temp_vis[temp_vis[c.name_sum]>0].sort_values(by=c.name_sum).plot(column=c.name_sum, ax=axes[1], cmap="Reds")
        temp_vis[temp_vis["dn_abs_pos"]>0].sort_values(by="dn_abs_pos").plot(column="dn_abs", ax=axes[2], cmap="RdBu_r", vmin=-0.5, vmax=0.5) # TODO: add colorbar?
        save_plot(f"{out_fname_id}__compare_dn_abs", dir_detail_plot)
    
    
        # sep. diff plot
        ## "dn_abs" refers to the absolute change in normalized centrality between c and cref  (dn_abs=0.1 for e.g. cref = 25% of max, c = 35% of max.) -> max. possible range is -1 to 1    
        
        ## first: fixed value range (cmap)
        ax = temp_vis.sort_values(by="dn_abs_pos").plot(column="dn_abs", cmap="RdBu", vmin=-0.2, vmax=0.2, linewidth=0.3)
        set_plot_aoi(ax, aoi_name, True, True, zoom_buffer=-5000)
        plot_map(ax, f"{out_fname_id}_dn_abs_colvalmax0.2", dir_detail_plot)
        
        # second: plot with colorbar, scaled to max. abs. diff.
        fig, ax = plt.subplots(1, 1)
        cb = plt.axes((.75,.16,.1,.02), facecolor="#000", alpha=0.5)
        cb.tick_params(labelsize=7)
        #cb.dividers.set_linewidth(2)
        temp_vis.sort_values(by="dn_abs_pos").plot(
            column="dn_abs", cmap="RdBu", vmin=-temp_vis.dn_abs_pos.max(), vmax=temp_vis.dn_abs_pos.max(), linewidth=0.3, figsize=(10, 10),
            ax=ax,
            legend=True,
            cax=cb, 
            legend_kwds={"orientation": "horizontal", "drawedges":False}, # "format": "{x:.1f}" # label
        )
        set_plot_aoi(ax, aoi_name, True, True, zoom_buffer=-5000)
        plot_map(ax, f"{out_fname_id}_dn_abs", dir_detail_plot)
        
        
        if plot_all:
            # plot centrality (ref) vs. centrality delta (norm)
            plt.figure(figsize=(5,5))
            plt.scatter(temp_vis[cref.name_sum].fillna(0)/temp_vis[cref.name_sum].max(), 
                        temp_vis["dn_abs"],
                        s=1, c="#017ABE")
            plt.xlabel(f"ref. ({cref_label})")
            plt.ylabel(f"{c_label}: normalized difference")
            save_plot(f"x_{out_fname_id}__cref_cdelta_norm", dir_detail_plot)
        
        # direct centr comparison
        plt.figure(figsize=(5,5))
        plt.scatter(temp_vis[cref.name_sum].fillna(0)/temp_vis[cref.name_sum].max(), 
                    temp_vis[c.name_sum].fillna(0)/temp_vis[c.name_sum].max(), 
                    s=1, c="#017ABE")
        plt.xlabel(f"ref. ({cref_label})")
        plt.ylabel(f"{c_label}")
        save_plot(f"x_{out_fname_id}__cref_scatter", dir_detail_plot)


    return {
        "aoi_name": aoi_name,
        "aoi_core_cseg_n": len(centr_df),
        "name_cref": cref.name_sum,
        "name_c": c.name_sum,
        "label_cref": cref_label,
        "label_c": c_label,
        "label_compare": ccomp_label,
        "c_type": c.type,
        "c_is_bp": c.is_bp,
        "c_dfac": c.dfac,
        "c_cut": c.cut,
        "c_nws": c.nws,
        "c_decay_from": c.decay_from,
        "par_q": c_quant,
        "par_p": c_ofmax,
        "trh_q": c_trh_q,
        "trh_p": c_trh_p,
        "d_max_c": d_max_c,
        "d_max_c_rel": d_max_c/centr_df[cref.name_sum].max(),
        "d_hc_seg_n": hc_seg_d_n,   # delta (count) of segments with hc (based on % of each max. centrality - "p" parameter)
        "d_hc_seg_n_rel": hc_seg_d_n/hc_seg_ref_n,
        "hc_seg_added": len(hc_seg_added),
        "hc_seg_removed": len(hc_seg_removed),
        "hc_seg_remained": n_hc_seg_remained,
        "hc_seg_share_changed": (len(hc_seg_added)+len(hc_seg_removed)) / (len(hc_seg_added) + len(hc_seg_removed) + n_hc_seg_remained),
        "hcq_d_mean": orig_hc_q.d_abs.mean(),
        "hcq_d_abs_mean": orig_hc_q.d_abs.abs().mean(),
        "hcq_dn_min": orig_hc_q.dn_abs.min(),
        "hcq_dn_mean": orig_hc_q.dn_abs.mean(),
        "hcq_dn_abs_mean": orig_hc_q.dn_abs.abs().mean(),
        "hcq_dn_max": orig_hc_q.dn_abs.max(),
        "hcq_dn_rel_mean": orig_hc_q.dn_rel.mean(),
        "hcp_d_mean": temp_vis[temp_vis[cref.name_sum] > c_trh_p].d_abs.mean(),
        "hcp_d_abs_mean": temp_vis[temp_vis[cref.name_sum] > c_trh_p].d_abs.abs().mean(),
        "hcp_dn_min": temp_vis[temp_vis[cref.name_sum] > c_trh_p].dn_abs.min(),
        "hcp_dn_mean": temp_vis[temp_vis[cref.name_sum] > c_trh_p].dn_abs.mean(),
        "hcp_dn_abs_mean": temp_vis[temp_vis[cref.name_sum] > c_trh_p].dn_abs.abs().mean(),
        "hcp_dn_max": temp_vis[temp_vis[cref.name_sum] > c_trh_p].dn_abs.max(),
        "hcp_dn_rel_mean": temp_vis[temp_vis[cref.name_sum] > c_trh_p].dn_rel.mean(),
        "d_mean": temp_vis.d_abs.mean(),
        "d_abs_mean": temp_vis.d_abs.abs().mean(),
        "dn_mean": temp_vis.dn_abs.mean(),
        "dn_abs_mean": temp_vis.dn_abs.abs().mean(),
        "dn_incr_gt10": sum(temp_vis.dn_abs > 0.1),
        "dn_decr_gt10": sum(temp_vis.dn_abs < -0.1),
        "dn_incr_gt20": sum(temp_vis.dn_abs > 0.2),
        "dn_decr_gt20": sum(temp_vis.dn_abs < -0.2),
        "dn_incr_gt50": sum(temp_vis.dn_abs > 0.5),
        "dn_decr_gt50": sum(temp_vis.dn_abs < -0.5),
        "n_turned_zero": len(centr_df[(centr_df[cref.name_sum] > 0) & (centr_df[c.name_sum] == 0)]),      # no. of segments that turned to zero centrality
        "n_turned_gt_zero": len(centr_df[(centr_df[cref.name_sum] == 0) & (centr_df[c.name_sum] > 0)])    # no. of segments that turned to greater than zero centrality
    }


def create_aoi_mask(aoi, outer, inner, target_crs=None):
    if target_crs is None:
        og = outer.loc[aoi, "geometry"].envelope
        ig = inner.loc[aoi, "geometry"]
    else:
        og = outer.to_crs(target_crs).loc[aoi, "geometry"].envelope
        ig = inner.to_crs(target_crs).loc[aoi, "geometry"]
    mask = og - ig 
    mask_gdf = gpd.GeoDataFrame([{"aoi_name": aoi, "geometry":mask}], geometry="geometry", crs=outer.crs)
    return mask_gdf

def plot_add_core_aoi(ax, aoi_name, core=None, facecolor="none", edgecolor="#B8014A", linewidth=0.5, zorder=100, custom_crs=None):
    if core is None:
        core = get_aoi_extent(aoi_name, get_full_extent=False, custom_crs=custom_crs)
    elif custom_crs is not None:
        core = core.to_crs(custom_crs)
    core.plot(ax=ax, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, zorder=zorder)

def plot_add_aoi_mask(ax, aoi_name, outer=None, inner=None, facecolor="#00000007", linewidth=0, zorder=99, custom_crs=None):
    if inner is None:
        inner = get_aoi_extent(aoi_name, get_full_extent=False, custom_crs=custom_crs)
    elif custom_crs is not None:
        inner = inner.to_crs(custom_crs)
    if outer is None:
        outer = get_aoi_extent(aoi_name, get_full_extent=True, custom_crs=custom_crs)
    elif custom_crs is not None:
        outer = outer.to_crs(custom_crs)
    create_aoi_mask(aoi_name,
                    outer=outer,
                    inner=inner,
                    ).plot(ax=ax, facecolor=facecolor, linewidth=linewidth, zorder=zorder)
        
def plot_set_extent(ax, bounds=None, gdf=None, index=None, crs=None, buffer:int=0):
    if bounds is None:
        if crs == None:
            bounds = gdf.loc[index].geometry.bounds
        else:
            bounds = gdf.to_crs(crs).loc[index].geometry.bounds
    ax.set_xlim([bounds[0]-buffer, bounds[2]+buffer])
    ax.set_ylim([bounds[1]-buffer, bounds[3]+buffer])
    
def set_plot_aoi(ax, aoi_name, plot_core_aoi=False, plot_aoi_mask=False, custom_crs=None, add_basemap=False, basemap_zoom_adjust=1, zoom_buffer:int=0):
    full = get_aoi_extent(aoi_name, get_full_extent=True, custom_crs=custom_crs)
    core = get_aoi_extent(aoi_name, get_full_extent=False, custom_crs=custom_crs)
    if plot_core_aoi:
        plot_add_core_aoi(ax, aoi_name, core, custom_crs=custom_crs)
    if plot_aoi_mask:
        plot_add_aoi_mask(ax, aoi_name, full, core)
    plot_set_extent(ax, gdf=full, index=aoi_name, buffer=zoom_buffer)
    if add_basemap:
        if custom_crs is None:
            custom_crs = full.crs
        cx.add_basemap(ax, crs=custom_crs, zoom_adjust=basemap_zoom_adjust, source=cx.providers.CartoDB.Positron)
    
def plot_map(ax, name, dir="plots", dpi=600, transparent=True):
    ## set plot options
    plt.margins(x=0, y=0)
    ax.set_axis_off()
    plt.savefig(os.path.join(dir, f"{name}.png"), dpi=dpi, bbox_inches='tight', pad_inches=0, transparent=transparent)
    plt.clf()
    plt.close()
    
    
def save_cdf(centr_df, file, x_samples=None):
    samples_optimized = [i/2000 for i in range(400)] # 20%
    samples_optimized += [0.2 + i/1000 for i in range(100)] # 30%
    samples_optimized += [0.3 + i/500 for i in range(351)] # 100%
    results = {"centr":samples_optimized}
    for cn in centr_df.columns:
        if not cn.startswith("centr_") or not cn.endswith("_sum"):
            continue
        norm = centr_df[cn].fillna(0) / centr_df[cn].max()
        cm = norm.value_counts().sort_index().cumsum()
        #x_samples =  [norm.index.max() * i/1000 for i in range(1001)]
        if x_samples is None:
            x_samples = samples_optimized
        x_sample_ids = cm.index.searchsorted(x_samples)
        sampled = cm.iloc[x_sample_ids].reset_index(drop=True)
        results = results | ({cn.rstrip("_sum"):sampled})
    df = pd.DataFrame(results)
    df.to_csv(file)
    df.set_index("centr", inplace=True)
    return df


    