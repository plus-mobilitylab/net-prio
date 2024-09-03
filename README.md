# net-prio

A method and software implementation for prioritizing road segments towards creating coherent cycling networks. Relying on open data, open source software, and focusing on the spatial perspective. The code included in this repository allows for systematic comparison of centrality (segment importance) variants, especially with regard to distance parameters.

## Method and implementation

The method is described in the following scientific paper, which also contains several examples and result assessment:
Werner, C., & Loidl, M. (2024). Creating coherent cycling networks: A spatial network science perspective *(submitted - will be updated soon)* [doi.org/10.5281/zenodo.13349332](https://doi.org/10.5281/zenodo.13349332)

The workflow is structured as follows:
- `1_compute_weights.py`: Computes spatial (and optionally population-based) node weights. These weights are then used as input for centrality computation)
- `2_compute_centrality.py`: Computes different variants of betweenness centrality (e.g. spatial betweenness centrality (SBC), population-weighted BC, standard edge BC)
- `3_a1_aois.ipynb`: Assesses the spatial extents (AOIs) and generates an overview map. Further creates a GeoPackage file containing geometries for the full AOI extents as well as the edge-effect-free core areas.
- `4_a2_plot_ba_and_c.ipynb`: Plots bikeability per AOI and importance/centrality results for the given centrality variants.
- `5_a3_compute_ps.ipynb`: Computes Priority Scores (several variants) for the defined centrality columns, based on importance (centrality) and bikeability (segment suitability).
- `6_a4_assess_ps.ipynb`: Assesses Priority Score results - i.e. histograms per bikeability class.
- `7_a5_assess_centr_distc.ipynb`: Comparative assessment of distance cutoff variants in centrality computation. 
- `8_a5_assess_centr_subsampling.ipynb`: Comparative assessment of node subsampling distance variants.
- `9_a6_assess_centr_net_simplification.ipynb`: Comparative assessment of network simplification.

In the following subsections we provide further detail on individual steps of the proposed method and the implementation.

### Computation of node weights

Computation of node weights is the first computational step and required for spatially normalized betweenness centrality and for population-weighted centrality.

- **spatial centrality** (SBC): Node weights are computed following the general concept of spatial centrality (Werner and Loidl, 2023). In addition, here, the tessellation polygons are intersected with built-up areas and specific land use types that are expected to typically generate trips (e.g. leisure areas such as parks, sports facilities, etc.). Plot types such as forests, crop land, etc. are excluded from contributing spatial weight. A comprehensive list of plot types considered is provided with the OpenStreetMap query in `overpass_area_query.txt`.
- **population-weighted centrality** (PBC): For PBC, the same tessellation polygons as for SBC are utilized. Instead of covered area in SBC, here, the population per area is used as weight. This is computed from population raster files (see requirements / data section).


### Network simplification

We developed an algorithm for network simplification which aims at reducing the impact on betweenness centrality results. Therefore, network simplification is executed individually per centrality variant (based on *node weight column* and *cost column*). 

For structural data compatibility of all centrality results, the workflow ensures to assign centrality results to the original input graph, while internally using the simplified graph for routing. 

Furthermore, the algorithm re-distributes the weights of removed nodes to their neighbours based on cost-weighted proximity.

Simplification targets two aspects:
- removal of dangling links (dead ends)
- joining subdivided links (removing nodes of `degree=4` in a directed graph)

By default, removal of dangling links is iteratively executed five times. Then, subdivided links are joined. This approach was found to be suitable for the typical scale of application. Further optimization may be conducted in the future, through fine-tuning the execution order and repetition count of both simplification methods.

You find the implementation in `algo/net_helper.py` with the function `net_simplify` forming the main entry point.


#### Removal of dangling links

Dangling links (or dead ends) are removed as follows:

- Self-loop edges are removed
- The graph is converted to an undirected graph
- Nodes with `degree=1` in the undirected graph are selected and removed from the original network
- The weight of each removed node is added to the node weight of its neighbor
- Adjacent edges of each removed node are also removed

(corresponding function: `_remove_ends()`)


#### Merging subdivided links

Candidates for merging are all nodes of `degree=4` (assuming a directed graph). This captures all nodes that are connected bi-directionally to two neighbor nodes. 
Candidate nodes are removed (and the adjacent segments joined) if
- they have zero weight and the option `always_merge_zero_w_nodes` is activated (default) 
- the length of the newly joined segment will not exceed the `seg_merge_threshold` distance (250 m by default)

In case of a segment join, the weight of the removed node is distributed to its neighbors inversely proportional to their cost distance (*cost column* value).

(corresponding function: `net_simplify_merge_segments()`)


### Option: node-subsampling based on regular grid

We optionally allow for sub-sampling origin and destination (OD) nodes by approximating a regular grid.
For compatibility with the workflow implementation, we do not directly sub-sample nodes from the network. 
Instead, we use the node weights data to merge all node weights within each grid cell to the node closest to the cell centroid.
Later in the workflow only nodes with `weight > 0` are considered as origin and destination for centrality computation.

(corresponding function: `nodes_weight_grid_sample()`)

### Betweenness Centrality algorithm

The algorithm for computing betweenness centrality is based on the spatial betweenness centrality implementation by Werner and Loidl (2023), which extends the SIBC implementation by Wu et al. (2022) that relies on NetworkX (Hagberg et al., 2008) as foundation.

The following extensions are added to the spatially normalized betweenness centrality approach:
- Distance cutoff: The implementation now allows to consider both euclidean network distance and cost distance (i.e. perceived "bikeable" distance) at the same time. This allows to define a maximum euclidean network distance as cutoff on route length, while routing based on perceived (bikeable path) distance.
- If using a distance cutoff, the network is spatially filtered for optimization. Routing then only uses the subgraph of nodes within the given maximum distance from a source node. Paths are later filtered according to Euclidean network distance and excluded if the cutoff is still exceeded.
- Implemented a parallelized version of the centrality algorithm.
- Added progress tracking (indicating current computation progress, elapsed time and estimated remaining time).


## Prerequisites

### Software

- ogr2ogr executable
- Python and the dependencies listed in `requirements.txt`
  We recommend installing the requirements using conda:
  e.g.: `conda create -n "net-prio" --file requirements.txt`
  (creates a new environment named 'net-prio' and installs the required dependencies)


### Data

In order to run the code provided in this repository, you need to provide the following files within the subdirectory `data_input`:

- **input network**: the GeoPackage file output of [NetAScore](http://github.com/plus-mobilitylab/netascore)

  - please **note**: For the recommended default settings (using `index_bike_incwalk`) please make sure to include the following lines in the NetAScore settings file (profiles section) to compute the bikeability index also for segments which are only accessible by foot:
    ```yaml
      -
        profile_name: bike_incwalk
        filename: profile_bike.yml
        filter_access_bike: True
        filter_access_walk: True
    ```

- *optional* - if you want to compute population-weighted centrality variants (PBC): provide population raster files in GeoTIFF format - e.g. [GHS-POP](https://human-settlement.emergency.copernicus.eu/download.php?ds=pop) - inside the subdirectory `data_input/population_raster`

## Output

During execution of the Python scripts and Jupyter notebooks several result files and intermediate output files are generated. These are stored within the `data` sub-directory.

Furthermore, plots are added by the workflow to the `plots` subdirectory.

Regarding naming conventions for files and GeoPackage columns please refer to the supplementary PDF file (which includes additional results not included in the paper as well as an introduction with a list of abbreviations).

## References

*Werner, C., & Loidl, M. (2024). Creating coherent cycling networks: A spatial network science perspective* *(submitted - will be updated soon)* [doi.org/10.5281/zenodo.13349332](https://doi.org/10.5281/zenodo.13349332)

*Werner, C., & Loidl, M. (2023). Betweenness Centrality in Spatial Networks: A Spatially Normalised Approach.* *In R. Beecham, J. A. Long, D. Smith, Q. Zhao, & S. Wise (Eds.), 12th International Conference on Geographic Information Science (GIScience 2023) (Vol. 277, p. 83:1-83:6). Schloss Dagstuhl – Leibniz-Zentrum für Informatik.* [doi.org/10.4230/LIPIcs.GIScience.2023.83](https://doi.org/10.4230/LIPIcs.GIScience.2023.83)

*Xiaohuan Wu, Wenpu Cao, Jianying Wang, Yi Zhang, Weijun Yang, and Yu Liu. A spatial interaction incorporated betweenness centrality measure. PLOS ONE, 17(5):e0268203, May 2022.* [doi.org/10.1371/journal.pone.0268203](https://doi.org/10.1371/journal.pone.0268203)

*Hagberg, A. A., Schult, D. A., & Swart, P. J. (2008). Exploring network structure, dynamics, and function using NetworkX. In G. Varoquaux, T. Vaught, & J. Millman (Eds.), Proceedings of the 7th python in science conference (pp. 11–15)*
