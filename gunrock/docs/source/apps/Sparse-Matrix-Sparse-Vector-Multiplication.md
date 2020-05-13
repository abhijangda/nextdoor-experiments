Overview
========

Given a sparse vector as the frontier, and the graph as the sparse matrix (edge weight as nnz values in the matrix), Gunrock can perform a sparse vector sparse matrix multiplication using an advance with reduce operation.

Per-Vertex and Per-Edge Data
============================

`// per-vertex`

`v_value[vid] // values on sparse vector`

`reduced_value[vid] //multiplication result`

`// per-edge`

`weight[eid] // nnz values in the dense matrix`

Functor and Operator
====================

The algorithm only needs an advance with reduce_by_key.

Advance: reduce_op is add, key is vertex ID.
Apply:
value_to_reduce = v_value[d_id] * weight[e_id]

Enactor
=======

f = Frontier.Init(sparse_vector)

Advance_with_Reduce(f, nullptr) // no output frontier