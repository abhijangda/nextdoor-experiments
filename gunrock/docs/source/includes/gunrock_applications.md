# Gunrock's Application Cases

The following is are a wide variety of graph primitives, including traversal-based (breadth-first search, single-source shortest path); node-ranking (HITS, SALSA, PageRank); and global (connected component, minimum spanning tree) implemented within Gunrock.

The "Directory" column in the table below shows, which subdirectory within `gunrock/app` and `examples` these applications' implementations are. The version number in the "Single GPU" and "Multi-GPU" columns show which API abstraction of gunrock supports the respective application. If you are interested in helping us port an application in the older abstraction (`v0.5.x`) to a newer, much cleaner abstraction (`v1.x.x`), please see our [Porting Guide](https://gunrock.github.io/docs/developers.html#porting_guide).


| Application                                     | Directory | Single GPU | Multi-GPU |
|-------------------------------------------------|-----------|------------|-----------|
| A* Search                                       | astar     | v0.5.x     |           |
| Betweenness Centrality                          | bc        | v1.x.x     | v0.5.x    |
| Breadth-First Search                            | bfs       | v1.x.x     | v1.x.x    |
| Connected Components                            | cc        | v1.x.x     | v0.5.x    |
| Graph Coloring                                  | color     | v1.x.x     |           |
| Geolocation                                     | geo       | v1.x.x     |           |
| RMAT Graph Generator                            | grmat     | v0.5.x     |           |
| Graph Trend Filtering                           | gtf       | v1.x.x     |           |
| Hyperlink-Induced Topic Search                  | hits      | v1.x.x     |           |
| K-Nearest Neighbors                             | knn       | v1.x.x     |           |
| Louvain Modularity                              | louvain   | v1.x.x     |           |
| Label Propagation                               | lp        | v0.5.x     |           |
| MaxFlow                                         | mf        | v1.x.x     |           |
| Minimum Spanning Tree                           | mst       | v0.5.x     |           |
| PageRank                                        | pr        | v1.x.x     | v0.5.x    |
| Local Graph Clustering                          | pr_nibble | v1.x.x     |           |
| Graph Projections                               | proj      | v1.x.x     |           |
| Random Walk                                     | rw        | v1.x.x     |           |
| GraphSAGE                                       | sage      | v1.x.x     |           |
| Stochastic Approach for Link-Structure Analysis | salsa     | v0.5.x     |           |
| Subgraph Matching                               | sm        | v1.x.x     |           |
| Shared Nearest Neighbors                        | snn       | v1.x.x     |           |
| Scan Statistics                                 | ss        | v1.x.x     |           |
| Single Source Shortest Path                     | sssp      | v1.x.x     | v0.5.x    |
| Triangle Counting                               | tc        | v1.x.x     |           |
| Top K                                           | topk      | v0.5.x     |           |
| Vertex Nomination                               | vn        | v1.x.x     |           |
| Who To Follow                                   | wtf       | v0.5.x     |           |
