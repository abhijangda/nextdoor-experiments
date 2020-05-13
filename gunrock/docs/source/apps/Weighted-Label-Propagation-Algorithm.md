Overview
========

Label propagation algorithm can be served as a way to detect communities in a directed/undirected graph. In a traditional label propagation algorithm, each node chooses the label occurring with the highest frequency among its neighbor list. This operation is very difficult to implement in parallel on the GPU. Instead, we use weighted label propagation, which assigns the label of the neighbor node that connects with the source node with the largest weighted edge. Reference paper: 
[Fast Community Detection Algorithm with GPUs and Multicore Architectures](http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=6012870&tag=1)

Edge Weight Assign
==================

edge weight can be assigned using the following equation:

weight(e(s,d)) = tc_count(e(s,d))/ sum_{k \in neighbor(s)} tc_count(e(s,k))

tc_count(e) is the number of triangles with one edge as e. This can be acquired by running a triangle counting and keep the tc_count for each edge, then do reduce by key according to source node ID.

Per-Vertex and Per-Edge Data
============================

`// per-vertex`

`label[vid] = vid // result community label (starting with vid)`

`reduced_vid[vid] = GR_INVALID_NODE_VAL //initialized as invalid`

`node_weight[vid]` = max(weight[e]) // max edge weights in neighborhood. can be computed via one pass of advance

`// per-edge`

`weight[eid] = compute_edge_weight() // pre-compute edge weights`

Primitives
==========
Initialize edges e(u,v) where node_weight[u]==node_weight[v] as active edge,
run connected component on the graph, update some label values.

Update Weights
==============
form a key value pair arrays where keys are labels, values are (degree, node weight) tuples.

W_l(L_i) = 1 - d_c/2M

W_l can be viewed as a regularizer for a specific label L_i (when more nodes in L_i, the impact gets lower)
d_c is the sum of degree for all nodes with label L_i. M is the edge number.

S_l(L_i) = \sum node_weight(u) where label[u] == L_i

These can be computed by segmented sort by labels as key and degree-weight tuple as value then reduce by key (label) or with current Gunrock operators, atomicAdd().

Functor and Operator
====================

The algorithm uses an advance with reduce_by_key and a parallel compute:

Advance: reduce_op is argmax of W_l(L_i)*S_l(L_i), key is vertex ID (cannot use labels directly as key since they are not adjacent).
{Cond,Apply}Edge both default one (keep all edges)

Compute Apply code (for all vertices):

`if (label[vid] != label[reduced_vid[vid]])`

`label[vid] = label[reduced_vid[vid]]`

Stop Condition
==============

reached max step or all labels are stable. This is the same as Pagerank.

Enactor
=======

`f = Frontier.Init(All_Edges)`

`while (!all_label_stable && iteration < max_step)`

`{`

  `UpdateWeights`

  `Advance_with_Reduce(f, f_out)`

  `Compute(f_out)`

`}`


