// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file hits_app.cuh
 *
 * @brief HITS Gunrock Application
 */

#include <gunrock/gunrock.h>

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>

// Graph definitions
#include <gunrock/graphio/graphio.cuh>
#include <gunrock/app/app_base.cuh>
#include <gunrock/app/test_base.cuh>

// HITS includes
#include <gunrock/app/hits/hits_enactor.cuh>
#include <gunrock/app/hits/hits_test.cuh>

namespace gunrock {
namespace app {
namespace hits {

template <typename ParametersT>
cudaError_t UseParameters(ParametersT &parameters);

/**
 * @brief Run hits tests
 * @tparam     GraphT        Type of the graph
 * @tparam     ValueT        Type of the distances
 * @param[in]  parameters    Excution parameters
 * @param[in]  graph         Input graph
...
 * @param[in]  target        where to perform the app
 * \return cudaError_t error message(s), if any
 */
template <typename GraphT>
cudaError_t RunTests(util::Parameters &parameters, GraphT &graph,
                     typename GraphT::ValueT *ref_hrank,
                     typename GraphT::ValueT *ref_arank,
                     util::Location target) {
  cudaError_t retval = cudaSuccess;

  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::ValueT ValueT;
  typedef typename GraphT::SizeT SizeT;
  typedef Problem<GraphT> ProblemT;
  typedef Enactor<ProblemT> EnactorT;

  // CLI parameters
  bool quiet_mode = parameters.Get<bool>("quiet");
  bool quick_mode = parameters.Get<bool>("quick");
  int num_runs = parameters.Get<int>("num-runs");
  double tol = parameters.Get<double>("tol");
  std::string validation = parameters.Get<std::string>("validation");
  util::Info info("HITS", parameters, graph);

  util::CpuTimer cpu_timer, total_timer;
  cpu_timer.Start();
  total_timer.Start();

  // Allocate problem specific host data
  ValueT *h_hrank = new ValueT[graph.nodes];
  ValueT *h_arank = new ValueT[graph.nodes];

  // Allocate problem and enactor on GPU, and initialize them
  ProblemT problem(parameters);
  EnactorT enactor;
  GUARD_CU(problem.Init(graph, target));
  GUARD_CU(enactor.Init(problem, target));

  cpu_timer.Stop();
  parameters.Set("preprocess-time", cpu_timer.ElapsedMillis());

  for (int run_num = 0; run_num < num_runs; ++run_num) {
    GUARD_CU(problem.Reset(target));
    GUARD_CU(enactor.Reset(graph.nodes, target));

    util::PrintMsg("__________________________", !quiet_mode);

    cpu_timer.Start();
    GUARD_CU(enactor.Enact());
    cpu_timer.Stop();
    info.CollectSingleRun(cpu_timer.ElapsedMillis());

    util::PrintMsg(
        "--------------------------\nRun " + std::to_string(run_num) +
            " elapsed: " + std::to_string(cpu_timer.ElapsedMillis()) +
            ", #iterations = " +
            std::to_string(enactor.enactor_slices[0].enactor_stats.iteration),
        !quiet_mode);

    if (validation == "each") {
      GUARD_CU(problem.Extract(h_hrank, h_arank));
      SizeT num_errors = Validate_Results(parameters, graph, h_hrank, h_arank,
                                          ref_hrank, ref_arank, false);
    }
  }

  cpu_timer.Start();

  GUARD_CU(problem.Extract(h_hrank, h_arank));

  if (validation == "last") {
    SizeT num_errors = Validate_Results(parameters, graph, h_hrank, h_arank,
                                        ref_hrank, ref_arank, tol, false);

    // num_errors stores how many positions are mismatched
    // Makes sense to keep this? Would need to sort first.
    if (!quiet_mode) {
      if (!quick_mode) {
        printf("CPU Algorithm Results:\n");
        DisplaySolution<GraphT>(ref_hrank, ref_arank, graph.nodes);
        printf("\n");
      }

      printf("GPU Algorithm Results:\n");
      DisplaySolution<GraphT>(h_hrank, h_arank, graph.nodes);
    }
  }

  // compute running statistics
  info.ComputeTraversalStats(enactor, (VertexT *)NULL);
#ifdef ENABLE_PERFORMANCE_PROFILING
#endif

  // Clean up
  GUARD_CU(enactor.Release(target));
  GUARD_CU(problem.Release(target));
  // Release problem specific data, e.g.:
  delete[] h_hrank;
  h_hrank = NULL;
  delete[] h_arank;
  h_arank = NULL;

  cpu_timer.Stop();
  total_timer.Stop();

  info.Finalize(cpu_timer.ElapsedMillis(), total_timer.ElapsedMillis());
  return retval;
}

}  // namespace hits
}  // namespace app
}  // namespace gunrock

// Above code remains from the original HITS test implementation. Below code is copied from sm_app.cuh.
// THIS NEEDS TO BE CHANGED


/*
 * @brief Entry of gunrock_sm function
 * @tparam     GraphT     Type of the graph
 * @tparam     ValueT     Type of the distances
 * @param[in]  parameters Excution parameters
 * @param[in]  graph      Input graph
 * @param[out] distances  Return shortest distance to source per vertex
 * @param[out] preds      Return predecessors of each vertex
 * \return     double     Return accumulated elapsed times for all runs
 */
template <typename GraphT, typename ValueT = typename GraphT::ValueT>
double gunrock_hits(
    gunrock::util::Parameters &parameters,
    GraphT &data_graph,
    ValueT *hub_ranks,
    ValueT *auth_ranks)
{
    typedef typename GraphT::VertexT VertexT;
    typedef gunrock::app::hits::Problem<GraphT  > ProblemT;
    typedef gunrock::app::hits::Enactor<ProblemT> EnactorT;
    gunrock::util::CpuTimer cpu_timer;
    gunrock::util::Location target = gunrock::util::DEVICE;
    double total_time = 0;
    if (parameters.UseDefault("quiet"))
        parameters.Set("quiet", true);

    // Allocate problem and enactor on GPU, and initialize them
    ProblemT problem(parameters);
    EnactorT enactor;
    problem.Init(data_graph, target);
    enactor.Init(problem   , target);

    problem.Reset(target);
    enactor.Reset(data_graph.nodes, target);

    cpu_timer.Start();
    enactor.Enact();
    cpu_timer.Stop();

    total_time += cpu_timer.ElapsedMillis();
    problem.Extract(hub_ranks, auth_ranks);
    

    enactor.Release(target);
    problem.Release(target);
    return total_time;
}

/*
 * @brief Simple interface take in graph as CSR format
 * @param[in]  num_nodes   Number of veritces in the input graph
 * @param[in]  num_edges   Number of edges in the input graph
 * @param[in]  row_offsets CSR-formatted graph input row offsets
 * @param[in]  col_indices CSR-formatted graph input column indices
 * @param[in]  edge_values CSR-formatted graph input edge weights
 * @param[in]  num_runs    Number of runs to perform SM
 * @param[out] subgraphs   Return number of subgraphs
 * \return     double      Return accumulated elapsed times for all runs
 */
template <
    typename VertexT = int,
    typename SizeT   = int,
    typename GValueT = float>
double hits(
    const SizeT        num_nodes,
    const SizeT        num_edges,
    const SizeT       *row_offsets,
    const VertexT     *col_indices,
    const int          num_iter,
    GValueT            *hub_ranks,
    GValueT            *auth_ranks)
{
    typedef typename gunrock::app::TestGraph<VertexT, SizeT, GValueT,
        gunrock::graph::HAS_CSR>
        GraphT;
    typedef typename GraphT::CsrT CsrT;

    // Setup parameters
    gunrock::util::Parameters parameters("hits");
    gunrock::graphio::UseParameters(parameters);
    gunrock::app::hits::UseParameters(parameters);
    gunrock::app::UseParameters_test(parameters);
    parameters.Parse_CommandLine(0, NULL);
    parameters.Set("graph-type", "by-pass");
    parameters.Set("max-iter", num_iter);
    bool quiet = parameters.Get<bool>("quiet");
    GraphT data_graph;

    // Assign pointers into gunrock graph format
    data_graph.CsrT::Allocate(num_nodes, num_edges, gunrock::util::HOST);
    data_graph.CsrT::row_offsets   .SetPointer((SizeT *)row_offsets, num_nodes + 1, gunrock::util::HOST);
    data_graph.CsrT::column_indices.SetPointer((VertexT *)col_indices, num_edges, gunrock::util::HOST);

    data_graph.FromCsr(data_graph.csr(), true, quiet);
    gunrock::graphio::LoadGraph(parameters, data_graph);

    // Run HITS
    double elapsed_time = gunrock_sm(parameters, data_graph, hub_ranks, auth_ranks);
    // Cleanup
    data_graph.Release();

    return elapsed_time;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
