# Graph Coloring as a Gunrock Operator
A graph $G=(V,E)$ is comprised of a set of vertices $V$ together with a set of edges $E$, where $E \subseteq V \times V$. Graph coloring $C: V \rightarrow N$ is a function that assigns a color  to each vertex that satisfies $C(v) \neq C(u)$ $\forall (v,u) \in E$. In other words, graph coloring assigns colors to vertices such that no vertex has the same color as a neighboring vertex.


Two characteristics of graph coloring that makes it an interesting operator are:

 1. Within a color, none of the vertices share an edge, and
 2. Each color represents a partitioning of vertices (or edges) of a whole graph.

### (1) Asynchrony
Asynchrony is the key benefit of using graph coloring as an operator, as explained above, within a single color, none of the vertices share an edge with each other. "Frog: Asynchronous Graph Processing on GPU with Hybrid Coloring Model" does a great job showing that exploring a hybrid-synchronous-asynchronous model can actually result is some benefit for primitives such as SSSP, PR, BFS and CC.

Within a single-GPU, this will benefit by avoiding block-wide communication, for a multi-GPU environment, we will avoid communicating across GPUs, and within a multi-node, we will attempt to minimize communication across nodes. The reason why this is a *hybrid* asynchronous model is because there is still a synchronization step (called color merging) at the very end where you finally consider other groups of colors which do have edges to each other. We would generally want to avoid this last step as much as possible, and choose applications (primitives) that coloring makes sense to do as an operator.

Yuechao and I, Muhammad, have a workshop paper called "Synchronous vs. Asynchronous GPU Graph Frameworks" analyzing Groute (asynchronous processing model) and Gunrock (bulk-synchronous model), and this approach will be a **hybrid-synchronous model**, and will be worth comparing against what we have already analyzed in the past paper.

Adding this *new* synchronization model also shows the programmability of Gunrock as a graph processing model, and how it can be further abstracted away to a fully asynchronous model, but still have capabilities or switches to go back to BSP where it makes sense. This work will also be a stepping stone to comparing to what Yuxin is working on, which is asynchronous graph algorithms.

### (2) Multi-Frontiers (Colored Frontiers)
Currently within Gunrock, we mainly use a single input frontier and a single output frontier to express all our primitives and applications. A frontier in Gunrock-world is a group of vertices or edges, in our case we would utilize the graph coloring inherent property of partitioning the graph into colors to generate and build on a single-frontier approach already present in Gunrock to do multi-frontiers (colored frontiers). In an initial approach, we can take a single frontier as an input, and generate multi-frontiers where each color represents a frontier.

In the Frog paper, each color (or frontier in Gunrock) is a separate kernel launch, and for very small frontiers they either rely on the CPU to do processing because the overhead of launching the kernel is more than the amount of parallel work available within it, or they just rely on GPU atomics (which according to them are not costly for a few operations -- I don't agree with this btw). In Gunrock's case, we won't be launching a new kernel for every frontier that is generated using graph coloring, instead we can rely on multi-frontier to be launched together in different schemes. These schemes will inherently answer the load-imbalance problem that is presented after we perform graph coloring:

1. **Threads/Warps/CTAs:** This way we can continue to utilize asynchrony, without worrying about the corner case where we have frontiers with too few nodes that we need to either do them atomically or on the CPU.
```
if (colored_frontier.num_nodes < X)
  launch work on Threads
if (colored_frontier.num_nodes > X && colored_frontier.num_nodes < Y)
  launch work on Warps
if (colored_frontier.num_nodes > Y && colored_frontier.num_nodes < Z)
  launch work on Blocks
```
2. Other gunrock load-balancing strategies (**LB** and **ALL_EDGES**) are still applicable to multi-frontier, by just considering each frontier the "sub-group" of a large frontier: https://github.com/gunrock/docs/blob/master/source/includes/load_balancing.md

3. **Load-balance and Color:** We can choose coloring heuristics such that the resulting frontiers we generate are already balanced. One of the example is considering CUDA today, each CUDA warp has 32 threads, and in the most simplest case we would like to assign each node to a thread within a warp with no communication to any other warps or threads outside of its own (asynchrony within a frontier), we can do coloring such that if a generated color contains 32 nodes, the next node to-be-placed within the same color will instead generate a new color for itself, therefore with a guarantee that each frontier must contain 32 nodes, along with satisfying the asynchrony condition. This can later be expanded to Cooperative Group (or symphonic CUDA), where a warp is not specifically defined to be 32 threads, it can be any arbitrary number. This is one of the many things we can try when coloring with **quality** instead of minimizing the color counts. This approach also removes the need to do load-balancing separately, the generated colors will be load-balanced.

## API Design (IMPORTANT)
Have to spend some time thinking about this, in my pov, this will be the hardest thing to do right. We need to be able to capture existing operators for a fuse operation, along with support ways to express algorithms asynchronously (due to coloring).
