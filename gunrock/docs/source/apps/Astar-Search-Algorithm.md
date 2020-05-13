Overview
========

A* search is a variation of Dijkstra's algorithm. It selects the path using the following equation:

f(node)=g(node)+h(node)

Where g(node) is the actual cost from source node to the current node, h(node) is the estimated cost from the current node to the destination node. When h(node) is 0, A* search degenerates to a normal Dijkstra's algorithm.

The heuristic cost estimation function is the most difficult part in A* search. For different applications, users should develop proper heuristic function.

Per-Vertex and Per-Edge Data
============================

`// per-vertex`

`g_cost[vid] = infinity // actual cost from source to node vid (initialized as infinity, for source node, 0 for source node)`

`f[vid] = infinity // final cost value used for priority queue split (initialized as infinity, 0 for source node)`

`pred[vid] = GR_INVALID_NODE_VAL // predecessor node id, initialized as invalid.`

`// per-edge`

`weight[eid] = weights // distance between u,v where e(u,v) is the edge`

Functor and Operator
====================

The algorithm uses an advance, a filter, and a bisect (for 2-level priority queue) or a multi-split (for multi-level priority queue) operators:

Advance:

CondEdge:

`new_g_cost = g_cost[s_id] + weight[e_id];`

`old_g_cost = atomicMin(g_cost[d_id], new_g_cost); // this will assign the min cost to g_cost`

`return (new_g_cost < old_g_cost);`

ApplyEdge:

`pred[d_id] = s_id`

Filter:

h = ComputeHeuristicFunc()

f[nid] = g_cost[nid] + h

Bisect:

Split the output frontier of filter into iteration_near_pile and iteration_far_pile by whether f[nid] > delta[level]. 
Append the iteration_far_pile at the end of global_far_pile queue.
If the iteration_near_pile is empty, bisect the global_far_pile queue according to delta[level+1].

Stop Condition
==============

Both current frontier and global_far_pile are empty (source node does not connect to destination node).
Reached destination node in the generated frontier.

Enactor
=======

`f = Frontier.Init(source_node)`

`far_pile = Frontier.Init()`

`while (!f.Contains(destination_node) && !f.IsEmpty() && !far_pile.IsEmpty())`

`{`

  `Advance(f,f_out)`

  `Filter(f_out,f)`

  `Bisect(f,far_pile,delta)`

`}`
