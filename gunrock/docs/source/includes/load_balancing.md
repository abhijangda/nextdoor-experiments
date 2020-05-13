# Gunrock's Load Balancing Policies              {#load_balancing}

While graph computation exhibits ample fine-grained parallelism, that parallelism is highly irregular. If we parallelize computation over vertices, for instance, neighboring vertices may have very different amounts of work. On a SIMD machine like a GPU, addressing this load imbalance is critical for performance.

The most significant low-level reason Gunrock demonstrates good performance on graph computation is its load-balancing implementation. Gunrock incorporates several different high-performance load-balancing policies. What are they and how do they work?

In this discussion we specifically consider an _advance_ operation on an input frontier of vertices that outputs a frontier of neighboring vertices. Informally, this operation asks "what are the neighboring vertices to the current frontier". Gunrock's _advance_ also supports input and/or output frontiers of edges, but in this document we focus on input and output frontiers of vertices.

## Terminology

### CTA

"CTA" is "cooperative thread array", synonymous with "thread block". Threads within a CTA can communicate/synchronize through (fast, local) shared memory. A CTA is a virtualized SM (streaming multiprocessor); the hardware scheduler assigns CTAs to SMs if the SM has enough resources to store and run the CTA.

### Neighbor list

Gunrock uses the term _neighbor list_ to describe the set of neighbors connected to a particular vertex. In pseudocode, the simplest advance operation is:

```
outputFrontier <- emptySet
forall vertex in inputFrontier:
  forall u in vertex.neighborList:
    outputFrontier.add(u)
```

### Tiles and grains

CTAs are assigned a certain number of input elements (e.g., vertices). We call that number a _grain_. Grains (of elements) are subdivided into _tiles_ (of elements). Generally, CTAs run one tile at a time (with a call named `ProcessTile()`), loading that tile of elements into shared memory, processing them, and outputting the results. The default tile size is 256 elements.

## Gunrock's Advance operator

The entry point to an _advance_ operation is the `gunrock::oprtr::Advance` call. For instance, SSSP has an `SSSPIterationLoop` struct, which contains a `Core` member function, which calls `oprtr::Advance`. `Advance` is templated by the operator type, which specifies the input and output types (vertices or edges). SSSP uses `oprtr::OprtrType_V2V`. `Advance` is also templated by the datatypes of its 6 arguments:

- The input graph
- The input frontier
- The output frontier
- The operator parameters
- The advance operation
- The filter operation

The first three are straightforward. The second three need a little explanation.

### Operator parameters

The operator parameters (`oprtr_parameters`) hold configuration information for a variety of user-settable parameters. The important parameter for this discussion is `advance_mode`, which specifies the load-balancing policy as a string, and is then used by the operator to choose its mode.

### Advance operation

The advance operation is a function that runs on a potential vertex in the output frontier. It returns true or false, and only if it returns true is that potential vertex placed in the output frontier. It can also modify data stored in the output vertex.

### Filter operation

The filter operation has identical behavior to the advance operation. It is used when Gunrock fuses an advance operator followed by a filter operator into a single fused operator, using this filter argument.

## Diving into Advance

The `Advance` call is in `oprtr/oprtr.cuh`. It simply calls `oprtr::advance::Launch`.

`Launch` is in `oprtr/advance/kernel.cuh`. It checks the `advance_mode` argument (a member of `oprtr_parameters`) and dispatches to the `Launch` call for that particular advance mode. The current advance modes are:

- `LB` ("load balanced")
- `LB_LIGHT` (a light variant of `LB`)
- `LB_CULL`("load balanced", with culling)
- `LB_LIGHT_CULL` (a light variant of `LB_CULL`)
- `TWC` (assigns vertices to one of "thread, warp, CTA")
    - The `ThreadExpand` load-balancing strategy is selected by configuring `TWC` to only assign vertices to threads.
- `ALL_EDGES`

In general, each of these strategies is separately implemented for `CSR` and `CSC` graph types. `Launch` calls `Launch_CSR_CSC` (in `oprtr/load_balance_strategy/kernel.cuh`), which calls `Kernel` (same file; this is a CUDA kernel call), which calls `PrepareQueue` (`oprtr/advance/advance_base.cuh`) then `Dispatch::Kernel` (same file). This last call is the core routine for implementing the load-balance strategy.

Now we look at the behavior of each of these load-balancing strategies.



## Static Workload Mapping Policy (`ThreadExpand`)

The simplest policy is to not attempt any load-balance across vertices at all. We assign each vertex in the input frontier to its own thread. Each thread then loads its assigned input vertex's neighbor list and serially iterates through this neighbor list. Gunrock calls this policy `ThreadExpand`.

The obvious disadvantage of this policy is that significant imbalance in work among neighboring vertices leads to poor performance. However, it has minimal load-balancing overhead, and if the input graph has a fairly uniform distribution of edges per vertex, it performs fairly well.

We don't use this policy in practice in Gunrock. It would be straightforward to write this as a standalone policy, or alternatively, in Gunrock, we can implement `ThreadExpand` by configuring the `TWC` strategy, as we will describe below.

<!--
yzh: "`ThreadExpand`, `WarpExpand` and `CtaExpand` together form both `TWC_FORWARD` and `TWC_BACKWARD` mode. You will see `ThreadExpand` appear in both `edge_map_forward/cta.cuh` and `edge_map_backward/cta.cuh`. `ThreadExpand()` maps to Merrill's `ExpandByScan()`. It uses each thread to put each node's neighbor list node ids to shared memory (https://github.com/gunrock/gunrock/blob/master/gunrock/oprtr/edge_map_forward/cta.cuh#L900). Then use one thread to expand several neighbor nodes (not necessarily from one input node) in a for loop: https://github.com/gunrock/gunrock/blob/master/gunrock/oprtr/edge_map_forward/cta.cuh#L907 "

there is no way to tell Gunrock "use `ThreadExpand` only" then.

yzh: "Actually there is a way. First you choose `TWC_FORWARD` mode, then here: https://github.com/gunrock/gunrock/blob/master/gunrock/app/bfs/bfs_enactor.cuh#L2020 You set `WARP_GATHER_THRESHOLD` to be std::numeric_limits<int>::max(), then none of the neighbor list will be considered as big enough to use `WarpExpand` or `CTAExpand`: https://github.com/gunrock/gunrock/blob/master/gunrock/oprtr/edge_map_forward/cta.cuh#L248 By setting `CTA_GATHER_THRESHOLD` to be the same or less than `WARP_GATHER_THRESHOLD`: , you can disable `WarpExpand()`: https://github.com/gunrock/gunrock/blob/master/gunrock/oprtr/edge_map_forward/cta.cuh#L501 By setting `CTA_GATHER_THRESHOLD` to be larger than the max possible neighbor list length, you can disable `CtaExpand()`: https://github.com/gunrock/gunrock/blob/master/gunrock/oprtr/edge_map_forward/cta.cuh#L390 "

yzh: "`work_limits.elements = grains_per_cta << LOG_SCHEDULE_GRANULARITY;` is the number of input elements processed per block. This number is further divided into tiles. Every ProcessTile() call will push the offset: `work_limits.offset += KernelPolicy::TILE_ELEMENTS`. So grain is another dimension of assigning input elements to CTAs. tile_size is how many input elements get processed together in a block. In our current settings, `TILE_ELEMENTS` is 256, meaning one `ProcessTile()` will expand 256 neighbor lists from input queue in a block. `grain_per_cta` is related to input queue length and block number for launching config."
-->

## Dynamic Workload Mapping Policy (`TWC`)

**Big picture: Assign small-sized neighbor lists to threads, medium-sized neighbor lists to warps, large-sized neighbor lists to CTAs (thread blocks). Process CTAs first, then warps, then threads.**

`ThreadExpand`'s disadvantage is its inability to load-balance across vertices with varying numbers of neighbors. Merrill and Garland's [seminal BFS implementation](http://research.nvidia.com/publication/scalable-gpu-graph-traversal) addresses this issue with its `TWC` ("thread-warp-CTA") policy. Gunrock's implementation is similar to Merrill/Garland's. As Merrill/Garland noted in their paper, this strategy can guarantee high utilization of resources and limit various types of load imbalance such as SIMD lane underutilization (by using per-thread mapping), intra-thread imbalance (by using warp-wise mapping), and intra-warp imbalance (by using block-wise mapping).

TWC groups neighbor lists into three categories based on their size, then individually processes each category with a strategy targeted directly at that size. Within Gunrock, we can set the thresholds for these three strategies, but the original Merrill/Garland work used 1) lists larger than a block, 2) lists larger than a warp (32 threads) but smaller than a block, and 3) lists smaller than a warp. These correspond to Gunrock's `CtaExpand()`, `WarpExpand()`, and `ThreadExpand()`. The programmer can manually set the thresholds for switch-points between these three strategies: `CTA_GATHER_THRESHOLD` is the breakpoint between the first two; `WARP_GATHER_THRESHOLD` is the breakpoint between the second two. If both are set to very large values, `TWC` devolves to `ThreadExpand`. The current default is `WARP_GATHER_THRESHOLD = 32` and `CTA_GATHER_THRESHOLD = 512`.

`Dispatch::Kernel` (in `oprtr/TWC_advance/kernel.cuh`)
first calls `GetCtaWorkLimits` (in `util/cta_work_distribution.cuh`), which fills a `work_limits` structure for that CTA with the number of grains and the starting grain for that CTA. Then the workhorse routine is `Cta::ProcessTile` in `oprtr/TWC_advance/cta.cuh`. It:

- Loads a tile (`LoadValid`) into registers. The intent with `LoadValid` is that data is thread-local unless it is needed across the block, in which case it will be moved to shared memory.
- Scans that tile (prefix-sum) to determine the number and distribution of  output elements (?)
- Runs `CtaExpand` with the `advance_op`. This loops until no thread has a neighbor list at least as long as `CTA_GATHER_THRESHOLD`.
    - All threads that have a neighbor list at least as large as `CTA_GATHER_THRESHOLD` attempt to get control of the CTA. One succeeds (the "command thread"). The command thread sets some block-wide constants that allow all threads to participate in the next phase.
    - Each thread then loads an edge from the command thread's neighbor list, then from it the remote vertex.
    - `ProcessNeighbor` (in `oprtr/advance/advance_base.cuh`) runs `advance_op` on that edge and if it passes `advance_op`, writes it to the output.
- Runs `WarpExpand` with the `advance_op`.
    - This is similar to `CtaExpand` except that one thread succeeds per warp, not per CTA, and the threshold is `WARP_GATHER_THRESHOLD`.
- Loops until done:
    - Runs `ThreadExpand`, which assembles a list of gather addresses (the vertex at the other end of the edge) in shared memory and fetches them (`GetEdgeDest`). As with the previous `Expand` calls, the loop calls `ProcessNeighbor` with `advance_op` to write vertices that pass `advance_op` to the output.

In general, this strategy does well with graphs with a high variance in vertex degree. However, it has relatively large overhead for two reasons: (1) it sequentially runs the CTA, Warp, and Thread expand methods, and (2) it may have load imbalance across CTAs, which we could address with work-stealing code but at additional cost.

## Load-Balanced Policy (`LB`)

TWC balances load during its execution, dynamically assigning one thread or group of threads to one vertex, with efficiency gains from choosing the right granularity of work to process each vertex or group of vertices. **Big picture: LB is a two-phase technique, with the first phase computing a perfect load balance (requiring device-wide computation to ensure this load balance) and the second phase actually performing the advance in a load-balanced way. LB can load-balance over the input or the output.** Generally this strategy is better for more even distributions of vertex degree.

<!-- YC: the tile and grain explanation should work for LB in addition to TWC, although it’s has not been done that way-->

<!-- todo MG suggestion: we have the ability to address load balance in several ways, including changing algorithm, doing runtime load balancing, and separating out irregular stuff from regular stuff and LB the irregular stuff -->

## Uniquify

<!-- John: "where do we have "uniquify" code in our load balancers, by which I mean "we get a bunch of vertices? as output, we remove all the duplicates, we probably do this at CTA granularity not globally"?"
YC: The uniquify part is in CULL_filter, https://github.com/gunrock/gunrock/blob/dev-refactor/gunrock/oprtr/CULL_filter/cta.cuh#L181 https://github.com/gunrock/gunrock/blob/dev-refactor/gunrock/oprtr/CULL_filter/cta.cuh#L315 and https://github.com/gunrock/gunrock/blob/dev-refactor/gunrock/oprtr/CULL_filter/cta.cuh#L344
they are in use when ENABLE_IDEMPOTENCE is true
otherwise filter will not remove the duplicates
The CULL_filter uses 4 different methods to find the duplicates: 1) bitmast: check a global bit mast, 2) Warp: check whether any thread in the warp has encountered the vertex recently; 3) History: check whether any thread in the block has encountered the vertex recently; 4) label: check a global per-element label. 2 & 3 use some kind of hash table to store the most recent vertices-->
