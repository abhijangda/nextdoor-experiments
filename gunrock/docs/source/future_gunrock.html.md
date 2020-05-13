---
title: Architectural thoughts on a future GPU graph framework

toc_footers:
  - <a href='https://github.com/gunrock/gunrock'>Gunrock&colon; GPU Graph Analytics</a>
  - Gunrock &copy; 2018 The Regents of the University of California.

search: true

full_length: true
---

# Architectural thoughts on a future GPU graph framework

We summarize thoughts from our group in terms of what a future GPU graph framework---Gunrock or a new framework---should have. We do not prioritize them here, nor rate their ease of implementation (though we could do either), but instead intend this to be a starting point for discussion.

Thanks to Yuechao Pan, Muhammad Osama, and Serban Porumbescu for input.

## Ease of programming

Our experience suggests that the *ease of programming a graph framework* is the most important aspect of its adoption. This is more important than raw performance. "Ease of programming" means many things, of course.

At the primitive level, it needs to be easy to input a graph and quickly ascertain performance on that graph. It should be able to be called/build standalone ("just run this primitive on this graph, how fast is it?") and it should be able to be linked into an existing application. It also needs to be easy to fit that primitive into a pipeline (what RAPIDS is doing).

Certainly many users will only run existing primitives (this is particularly true for, say, NetworkX users who want to migrate to GPUs with as little effort as possible). However, the breadth of graph applications that we saw as part of HIVE, and the fraction of those graph applications that were relatively minor modifications to existing primitives, makes me feel that a successful graph framework needs to be programmable at a lower level as well. Users want to write their own primitives, or tweak existing primitives.

It is of course sensible to begin with an interface that is primitive-only (as nvGRAPH did), but that cannot be the long-term plan. I understand the challenge: exposing a lower-level interface before it is mature possibly burdens the framework developer with supporting that obsolete interface forever. So not doing it initially is sensible. But it must be part of the roadmap.

Exposing this is also vital for research purposes; a graph framework with entry points only at the primitive level is not suitable for research.

In the long term, a graph framework should have *multiple entry points*: at least the primitive level and the operator level. These should not be hidden APIs. The framework must have a well-structured hierarchy with stable entry points at each level of the hierarchy (or at least if there are changes to the API, it's straightforward to move from the old to the new API).

## Push and pull are first-class

Our experience with the MIT GraphIt team and MIT professor Julian Shun's research before arriving at MIT make us feel that push and pull must be first-class ingredients of any future framework. From an API point of view, push and pull are straightforward; it's the underlying implementations and how they can quickly make decisions to switch and actually switch between push and pull that is the challenge. *Operators should fully support push and pull operation as one of their primary tasks, and the APIs should easily support this.*

## Separable components

My biggest architectural regret from Gunrock was that we did not design it from the outset into different modules. It was not clear at the time what those modules would be, however, but our lessons from that will hopefully be useful for future frameworks. They should include:

### Load balancing

We believe that good load balancing is the most important low-level reason for Gunrock's performance. We also see that load balancing is a critical technology for other sparse problems (e.g., sparse linear algebra, and for future sparse machine learning problems).

"Load balancing" is an amorphous term and it is not yet clear what that API should look like. We are actively researching this problem. The challenge is developing a high-performance separable load-balancing module that has both a clean interface+abstraction and delivers top performance. That interface+abstraction also needs to address other load-balancing problems.

### Data structure access

Gunrock originally supported raw CSR as an input format and all accesses into graph data structures assumed CSR. We have refactored Gunrock to put in a thin abstraction layer that is more functional so that we can explore mutable graph data structures. This is a bottom-up approach; we hope that we have captured the operations we want to do on a data structure (e.g., "for this vertex, return all its neighbors"), but this will take more work on our part.

Future frameworks should begin with a separable data-structure abstraction that assumes nothing about the data structure's implementation. We expect a future framework will support CSR/CSC/COO; various mutable data structures; and a "precompiled" data structure that is in a custom, framework-specific framework that yields the best performance. (For instance, usually our load balancing efforts do some prefix sums to allocate work. Those efforts would already be completed in the custom format. This would yield the best performance at the cost of generality and may be useful in performance-critical applications, or where graphs are computed once and used many times.)

Industry has invested significant effort into database-based graph representations. A data-structure API should support a database as the "data structure".

### Partitioning

Gunrock chose to separate partitioners from the rest of the implementation. This would be a good choice for future frameworks. It made our investigations easier and of course reduced complexity.

### Applications

Gunrock's applications (and primitives) are separated from Gunrock's core, which was a good decision. One component of this is a "test driver" that can run our gtest tests. We appreciate the benefit of having standalone programs (that are automatically tested on checkins) that exercise Gunrock's application and core code. They are also excellent examples to show how Gunrock works and a good entry point for new Gunrock developers.

### More?

We haven't put a lot of thought into what a completely factored Gunrock would look like, but this exercise is well worth doing.

## Different front ends/back ends

We believe it is important to support different front ends and back ends, that the APIs should support this cleanly. Which front or back ends is not the important part but instead that APIs should support input from multiple front ends and output to multiple back ends.

Front ends might include NetworkX and other python-like interfaces (at the interface level) and certainly GraphIt. Query languages like SPARQL or Cypher are also interesting.

Back ends should certainly include a Graph BLAS implementation (perhaps one custom implementation and one based on cuBLAS), as well as a native graph implementation (like Gunrock). We note we could *output* to a GraphIt back end (on a CPU, for instance) as well. NVIDIA would probably not be interested in an output to a different (non-GPU) architecture, but this is also an option.

## Good performance on CSR inputs

We strongly believe that Gunrock's emphasis on inputting standard formats like CSR is the right one. We expect Gunrock will largely be used as an intermediate stage in a pipeline where the input dynamically comes from a previous stage (say, a database) and output goes to another stage (say, visualization). CSR or other standard formats are critical to allow these pipelines to be built (this is congruent with the Arrow philosophy).

We see many papers that "beat" Gunrock in performance by choosing their own custom data structures, which essentially moves in-Gunrock computation (which we count in our runtimes) to precomputation of data structures (which other frameworks do not). However, these custom data-structure formats are not useful in a pipeline scenario. I believe they should be lower priority than standard Arrow sparse input formats.

## Python/high-level functional model for prototyping

During the preparation of HIVE applications, our consultant Ben Johnson wrote a simple Python application (`pygunrock`) that he used to rapidly prototype Gunrock development. This was at the operator level: advance, filter, compute. It simply allowed a developer to quickly confirm that a high-level Gunrock program was going to do the right thing. `pygunrock` is not mature.

Because writing a full (C++) Gunrock program is rather complex, we believe this prototyping environment is quite worthwhile.

## Fine-grained interoperability with arbitrary GPU computation

For me, the biggest surprise from our HIVE application development (from seeing a dozen new applications that we had not seen before) was the need for fine-grained interoperability between Gunrock and arbitrary computation.

The "standard" graph applications that the GPU computing community has used are the ones we all know: BFS, PageRank, SSSP, triangle counting, connected components, betweenness centrality. What we now realize about these applications is that they are focused almost exclusively on graph *operations* with very little interesting *computation*. (BFS, for instance, only has the math operation "add one to the depth".)

The more complicated applications we saw in HIVE have much more interesting computation. So where we previously thought we would run graph operations (advance, filter, etc.) with an occasional simple compute step, we are instead seeing more interesting computational steps, full kernels and/or calls into libraries (e.g., linear algebra). These computational steps need to be lightweight because they are more frequent; they can't have a high setup cost. Most importantly, the abstraction for writing programs in the graph framework needs to recognize the need for fine-grained interoperability, for quickly switching between compute and graph steps.

## Coarse-grained interoperability with other packages through Arrow/equivalent

Arrow and RAPIDS both recognize that graph components will interoperate with other non-graph components in larger pipelines. Gunrock is architecturally suited for such a model, although to my knowledge Arrow+GPU development has primarily concentrated on dense 2D dataframes and not on the sparse data structures that we need.

The important point about the design of this capability into a future graph framework is that we want to abstract away the details of the data structures used for communication. Certainly at first that may only be a single data structure (e.g., CSR), but the design cannot be closely coupled to that decision; we expect more (not a lot more, but definitely more) data structures in the future that are used to interoperate with other components of a pipeline.

## Optimization of sequence of graph computations

As graph programs get more complex, we expect that we will see programmers produce non-optimal graph computations. This will be for two reasons: one, those programmers are not experienced; two; programmers will use pieces/subroutines from other programs/abstractions and when all those pieces are stitched together, they will be non-optimal.

This situation is analogous to machine-learning frameworks (e.g., TensorFlow), who input a dataflow graph and whose frameworks then optimize that graph (both generically and for a particular back end). The ML framework takes responsibility from the programmer for some of optimizations.

For both compile-time and runtime optimizations (below), the important architectural decision in the near term is to include the ability to do both compile-time and runtime optimizations into the framework as first-class concepts, even if the initial capabilities of those optimizations is modest.

### Compile-time

One optimization in Gunrock is fusion of compute operators with both filter and advance operators. Logically, for example, the programmer writes a compute operator then a filter (which would normally entail two kernel calls); Gunrock fuses those into one kernel. This is currently explicit in the Gunrock programming model. A compile-time framework, however, could recognize that pattern and fuse it automatically without the programmer having to specify it. ([IrGL](https://dl.acm.org/citation.cfm?id=2984015) is an example of this approach.)

### Runtime

The ideal framework would also make runtime optimizations. For instance, direction-optimized BFS switches from push to pull during the computation. This switch could be guided by a runtime framework (that is more sophisticated than what we currently do, which is write custom code that does this switch only for this particular primitive). It may recognize, for instance, that a particular input graph is better suited for a particular load-balancing strategy or a Graph BLAS back end.

## Interesting features to explore

We have a long list of features that we think are worthwhile to explore in future frameworks. Understanding what might be desirable in the future is important now to make sure that APIs and abstractions allow the simple integration of these features into a framework if they are shown to be beneficial.

### Compression

We expect many multi-GPU graph applications are going to be inter-GPU bandwidth-limited, particularly when multiple nodes are involved. Gunrock does no compression (and I'm not aware of any other GPU frameworks that do) but CPU frameworks have been successful at even simple schemes. We believe it is sensible to build in API support for compression even if compression is not yet used.

### Remapping

Another win that we have seen in CPU frameworks has been remapping vertices and edges with different vertex/edge IDs. This potentially allows more locality in memory access and allows using fewer bits to represent, say, a frontier of vertices/edges.

### Asynchrony

Gunrock is currently a bulk-synchronous framework across multiple GPUs. Especially when moving to multiple nodes, this may not be the right approach for performance. Asynchronous frameworks (e.g., [Groute](https://dl.acm.org/citation.cfm?id=3018756)) can potentially alleviate some of the costs of having a global barrier across all GPUs. They are, however, a real challenge to program.

Future frameworks should have the ability to communicate asynchronously, even if that capability is not enabled at first.

### Mutability

We continue to research mutable data structures. Data structure APIs should be designed to support mutations. Performance is obviously important, but also identifying the needs of application developers with respect to mutability (e.g., can we separate data structure accesses into two phases, "insert/delete" and "query"?) is also critical.
