# gap-detection-priorities



## Method and implementation

The method is outlined in the following scientific paper:
*(work in progress - will be updated soon)*

The workflow is structured as follows:
- `1_compute_weights.py`: Compute spatial and population-based node weights (which are used as input for centrality computation)
- `2_compute_ca_cent.ipynb` (optional): extract POIs (railway stations in the example) and compute centrality-based accessibility metrics for these POIs
- `3_compute_full_cent.py`: Compute different variants of betweenness centrality
- `gaps-priorities.ipynb`: Generate priority scores based on segment-scale bikeability and segment importance (centrality-based)

In the following subsections we provide further detail on method and implementation for the individual steps of the proposed method.

### Computation of node weights

<TODO: brief intro, refer to spatial betweenness centrality paper, state differences / extensions - especially the spatial filtering for built-up and leisure areas>


### Network simplification

We developed an algorithm for network simplification which aims at reducing the impact on betweenness centrality results. Therefore, network simplification is executed individually per centrality variant (based on *node weight column* and *cost column*). 

For data compatibility of centrality results, the workflow ensures to assign centrality results to the original input graph, while internally using the simplified graph for routing. 

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

We optionally allow for sub-sampling nodes by approximating a regular grid.
For compatibility with the workflow implementation, we do not directly sub-sample nodes from the network. 
Instead, we use the node weights data to merge all node weights within each grid cell to the node closest to the cell centroid.
Later in the workflow only nodes with `weight > 0` are considered for centrality computation.

(corresponding function: `nodes_weight_grid_sample()`)


### Betweenness Centrality algorithm

<TODO>
origin: SBC paper + X

extensions:
- dist cutoff + cost cutoff (tracking both distance / cost values simultaneously)
- (can use bikeable paths routing (cost-based), but still restrict Euclidean path length using dist cutoff)
- if dist cutoff applied: spatial filter for optimization: use subgraph (spatial buffer on nodes); paths are later filtered according to Euclidean path length and excluded if the cutoff is exceeded
- implemented parallelized version of centrality algorithm
- added progress tracking (indicating progress, elapsed and estimated remaining time)


### POI-based centrality / accessibility

<TODO>


### Full network centrality

<TODO>


## Prerequisites

### Software

- ogr2ogr executable
<TODO: add all software and package dependencies - export conda env>


### Data

In order to run the code provided in this repository, you need to provide the following files:

- input network (output of NetAScore, using the following settings (include bike_incwalk) <TODO>)
- optional - if you want to compute population-weighted centrality variants (pbc): population raster - e.g. (GHS-POP)[https://human-settlement.emergency.copernicus.eu/download.php?ds=pop] (place inside `data_input/population_raster`)
<TODO: anything else?>

Exemplary data for Salzburg can be downloaded from here: <TODO: link to data repo (Zenodo)>

Please place all required files within the `data_input` sub-directory.