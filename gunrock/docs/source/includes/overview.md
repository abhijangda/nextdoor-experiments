# Overview

## What is _Gunrock_?
Gunrock is a stable, powerful, and forward-looking substrate for GPU-based graph-centric research and development. Like many graph frameworks, it leverages a bulk-synchronous programming model and targets iterative convergent graph computations. We believe that today Gunrock offers both the best performance on GPU graph analytics as well as the widest range of primitives.

## Who may use _Gunrock_?

+ **External Interface Users:** Users interested in leveraging the external C, C++ and/or Python interfaces to call high-performant applications and primitives (such as Breadth First Search, Connected Components, PageRank, Single Source Shortest Path, etc.) within Gunrock to perform graph analytics.

+ **Application Developers:** Uses interested in developing applications, primitives and/or low level operators for Gunrock.

+ **Graph Analytics Library Developers:** (CUDA and/or other languages -- for more backends, etc.)

## Why use _Gunrock_?
-   **Gunrock has the best performance of any programmable GPU+graph library.** Gunrock primitives are an order of magnitude faster than  (CPU-based) Boost, outperform any other programmable GPU-based system, and are comparable in performance to hardwired GPU graph primitive implementations. When  compared to [Ligra](https://github.com/jshun/ligra), a best-of-breed CPU system, Gunrock currently  matches or exceeds Ligra's 2-CPU performance with only one GPU.

    Gunrock's abstraction separates its programming model from the  low-level implementation details required to make a GPU  implementation run fast. Most importantly, Gunrock features very powerful load-balancing capabilities that effectively address the   inherent irregularity in graphs, which is a problem we must address   in all graph analytics. We have spent significant effort developing   and optimizing these features---when we beat hardwired analytics,   the reason is load balancing---and because they are beneath the   level of the programming model, improving them makes all graph   analytics run faster without needing to expose them to programmers.

-   **Gunrock's data-centric programming model is targeted at GPUs and offers advantages over other programming models.** Gunrock is written in a higher-level abstraction than hardwired implementations, leveraging reuse of its fundamental operations across different graph primitives. Gunrock has a bulk-synchronous programming model that operates on a frontier of vertices or edges; unlike other GPU-based graph analytic programming models, Gunrock focuses not on sequencing *computation* but instead on sequencing *operations on frontier data structures*. This model has two main operations: *compute*, which performs a computation on every element in the current frontier, and *traversal*, which generates a new frontier from the current frontier. Traversal operations include *advance* (the new frontier is based on the neighbors of the current frontier) and *filter* (the new frontier is a programmable subset of the current frontier). We are also developing new Gunrock operations on frontier data structures, including neighbor, gather-reduce, and global operations.

    This programming model is a better fit to high-performance GPU implementations than traditional programming models adapted from CPUs. Specifically, traditional models like gather-apply-scatter (GAS) map to a suboptimal set of GPU kernels that do a poor job of capturing producer-consumer locality. With Gunrock, we can easily integrate compute steps within the same kernels as traversal steps. As well, Gunrock's frontier-centric programming model is a better match for key optimizations such as push-pull direction-optimal search or priority queues, which to date have not been implemented in other GPU frameworks, where they fit poorly into the abstraction.

-   **Gunrock supports more primitives than any other programmable GPU+graph library.** We currently support a wide variety of graph primitives, including traversal-based (breadth-first search, single-source shortest path); node-ranking (HITS, SALSA, PageRank); and global (connected component, minimum spanning tree). Many more algorithms are under active development, see [Gunrock Applications](https://gunrock.github.io/docs/#gunrock-39-s-application-cases).

-   **Gunrock has better scalability with multiple GPUs on a node than any other graph library.** We not only show [better BFS performance on a single node than any other GPU framework](http://arxiv.org/abs/1504.04804) but also outperform other frameworks, even those customized to BFS, with up to four times as many GPUs. More importantly, our framework supports all Gunrock graph primitives rather than being customized to only one primitive.

-   **Gunrock's programming model scales to multiple GPUs while still using the same code as a single-GPU primitive.** Other frameworks require rewriting their primitives when moving from one to many GPUs. Gunrock's multi-GPU programming model uses single-node Gunrock code at its core so that single-GPU and multi-GPU operations can share the same codebase.
