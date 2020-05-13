// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * hello_problem.cuh
 *
 * @brief GPU Storage management Structure for hello Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>

#define N_HOPS 2

namespace gunrock {
namespace app {
// <TODO> change namespace
namespace hello {
// </TODO>

/**
 * @brief Speciflying parameters for hello Problem
 * @param  parameters  The util::Parameter<...> structure holding all parameter
 * info \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_problem(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;

  GUARD_CU(gunrock::app::UseParameters_problem(parameters));

  // <TODO> Add problem specific command-line parameter usages here, e.g.:
  // GUARD_CU(parameters.Use<bool>(
  //    "mark-pred",
  //    util::OPTIONAL_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
  //    false,
  //    "Whether to mark predecessor info.",
  //    __FILE__, __LINE__));
  // </TODO>

  return retval;
}

/**
 * @brief Template Problem structure.
 * @tparam _GraphT  Type of the graph
 * @tparam _FLAG    Problem flags
 */
template <typename _GraphT, ProblemFlag _FLAG = Problem_None>
struct Problem : ProblemBase<_GraphT, _FLAG> {
  typedef _GraphT GraphT;
  static const ProblemFlag FLAG = _FLAG;
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::ValueT ValueT;
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::CsrT CsrT;
  typedef typename GraphT::GpT GpT;

  typedef ProblemBase<GraphT, FLAG> BaseProblem;
  typedef DataSliceBase<GraphT, FLAG> BaseDataSlice;

  // ----------------------------------------------------------------
  // Dataslice structure

  /**
   * @brief Data structure containing problem specific data on indivual GPU.
   */
  struct DataSlice : BaseDataSlice {
    // <TODO> add problem specific storage arrays:
    util::Array1D<SizeT, VertexT> neighbors[N_HOPS];
    util::Array1D<SizeT, SizeT> lengths[N_HOPS];
    util::Array1D<SizeT, SizeT> positions[N_HOPS];
    util::Array1D<SizeT, SizeT> total_lengths[N_HOPS];
    int neighbors_lengths[N_HOPS];

    // </TODO>

    /*
     * @brief Default constructor
     */
    DataSlice() : BaseDataSlice() {
      // <TODO> name of the problem specific arrays:
      for (int i = 0; i < N_HOPS; i++) {
        neighbors[i].SetName("neighbors_" + std::to_string (i));
        lengths[i].SetName("lengths_"+std::to_string (i));
        positions[i].SetName("positions_"+std::to_string (i));
        total_lengths[i].SetName("total_lengths_"+std::to_string (i));
      }
      // </TODO>
    }

    /*
     * @brief Default destructor
     */
    virtual ~DataSlice() { Release(); }

    /*
     * @brief Releasing allocated memory space
     * @param[in] target      The location to release memory from
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Release(util::Location target = util::LOCATION_ALL) {
      cudaError_t retval = cudaSuccess;
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx));

      // <TODO> Release problem specific data, e.g.:
      for (int i = 0; i < N_HOPS; i++) {
        GUARD_CU(neighbors[i].Release(target));
        GUARD_CU(lengths[i].Release(target));
        GUARD_CU(positions[i].Release(target));
        GUARD_CU(total_lengths[i].Release(target));
      }
      // </TODO>

      GUARD_CU(BaseDataSlice ::Release(target));
      return retval;
    }

    /**
     * @brief initializing sssp-specific data on each gpu
     * @param     sub_graph   Sub graph on the GPU.
     * @param[in] gpu_idx     GPU device index
     * @param[in] target      Targeting device location
     * @param[in] flag        Problem flag containling options
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Init(GraphT &sub_graph, int num_gpus, int gpu_idx,
                     util::Location target, ProblemFlag flag) {
      cudaError_t retval = cudaSuccess;

      GUARD_CU(BaseDataSlice::Init(sub_graph, num_gpus, gpu_idx, target, flag));
      
      // <TODO> allocate problem specific data here, e.g.:
      for (int i = 0; i < N_HOPS; i++)
      {
        neighbors_lengths[i] = -1;
        GUARD_CU(positions[i].Allocate(sub_graph.nodes, target));
        GUARD_CU(lengths[i].Allocate(sub_graph.nodes, target));
        GUARD_CU(total_lengths[i].Allocate(1, target));
      }
      //GUARD_CU(neighbors.Allocate(sub_graph.edges, target));
      // </TODO>

      if (target & util::DEVICE) {
        // <TODO> move sub-graph used by the problem onto GPU,
        GUARD_CU(sub_graph.CsrT::Move(util::HOST, target, this->stream));
        // </TODO>
      }
      return retval;
    }

    cudaError_t InitNeighborsForHop (SizeT neighbors_length, int hop, util::Location target) {
      cudaError_t retval = cudaSuccess;
      GUARD_CU(neighbors[hop].Allocate(neighbors_length, target));
      return retval;
    }
    /**
     * @brief Reset problem function. Must be called prior to each run.
     * @param[in] target      Targeting device location
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Reset(util::Location target = util::DEVICE) {
      cudaError_t retval = cudaSuccess;
      SizeT nodes = this->sub_graph->nodes;
      SizeT edges = this->sub_graph->edges;

      // Ensure data are allocated
      // <TODO> ensure size of problem specific data:
      for (int i = 0; i < N_HOPS; i++) {
        GUARD_CU(lengths[i].EnsureSize_(nodes, target));
        GUARD_CU(positions[i].EnsureSize_(nodes, target));
        //GUARD_CU(neighbors[i].EnsureSize_(edges, target));
        GUARD_CU(total_lengths[i].EnsureSize_(1, target));
      }
      // </TODO>

      // Reset data
      // <TODO> reset problem specific data, e.g.:
      for (int i = 0; i < N_HOPS; i++) {
        GUARD_CU(lengths[i].ForEach([] __host__ __device__(SizeT &x){x = 0;}, nodes, target, this->stream));
        GUARD_CU(positions[i].ForEach([] __host__ __device__(SizeT &x){x = 0;}, nodes, target, this->stream));
        GUARD_CU(total_lengths[i].ForEach([] __host__ __device__(SizeT &x){x = 0;}, 1, target, this->stream));
      }

      //GUARD_CU(neighbors.ForEach([] __host__ __device__(VertexT &x) { x = -1; },
      //edges, target, this->stream));
      // </TODO>

      return retval;
    }
  };  // DataSlice

  // Set of data slices (one for each GPU)
  util::Array1D<SizeT, DataSlice> *data_slices;

  // ----------------------------------------------------------------
  // Problem Methods

  /**
   * @brief hello default constructor
   */
  Problem(util::Parameters &_parameters, ProblemFlag _flag = Problem_None)
      : BaseProblem(_parameters, _flag), data_slices(NULL) {}

  /**
   * @brief hello default destructor
   */
  virtual ~Problem() { Release(); }

  cudaError_t InitNeighborsForHop (SizeT neighbors_length, int hop, util::Location target = util::LOCATION_ALL) {
    cudaError_t retval = cudaSuccess;
    for (int i = 0; i < this->num_gpus; i++)
      GUARD_CU(data_slices[i][0].InitNeighborsForHop(neighbors_length, hop, target));
    return retval;
  }
  /*
   * @brief Releasing allocated memory space
   * @param[in] target      The location to release memory from
   * \return    cudaError_t Error message(s), if any
   */
  cudaError_t Release(util::Location target = util::LOCATION_ALL) {
    cudaError_t retval = cudaSuccess;
    if (data_slices == NULL) return retval;
    for (int i = 0; i < this->num_gpus; i++)
      GUARD_CU(data_slices[i].Release(target));

    if ((target & util::HOST) != 0 &&
        data_slices[0].GetPointer(util::DEVICE) == NULL) {
      delete[] data_slices;
      data_slices = NULL;
    }
    GUARD_CU(BaseProblem::Release(target));
    return retval;
  }

  /**
   * @brief Copy result distancess computed on GPUs back to host-side arrays.
...
   * \return     cudaError_t Error message(s), if any
   */
  cudaError_t Extract(
      // <TODO> problem specific data to extract
      int hop,
      VertexT *h_neighbors,
      int h_neighbors_length,
      SizeT *h_positions,
      SizeT *h_lengths,
      SizeT *h_total_lengths,
      // </TODO>
      util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    SizeT nodes = this->org_graph->nodes;
    SizeT edges = this->org_graph->edges;

    if (this->num_gpus == 1) {
      auto &data_slice = data_slices[0][0];

      // Set device
      if (target == util::DEVICE) {
        GUARD_CU(util::SetDevice(this->gpu_idx[0]));

        // <TODO> extract the results from single GPU, e.g.:
        GUARD_CU(data_slice.neighbors[hop].SetPointer(h_neighbors, h_neighbors_length*sizeof(VertexT), util::HOST));
        GUARD_CU(data_slice.neighbors[hop].Move(util::DEVICE, util::HOST));
        GUARD_CU(data_slice.lengths[hop].SetPointer(h_lengths, nodes*sizeof (SizeT), util::HOST));
        GUARD_CU(data_slice.lengths[hop].Move(util::DEVICE, util::HOST));
        GUARD_CU(data_slice.positions[hop].SetPointer(h_positions, nodes*sizeof (SizeT), util::HOST));
        GUARD_CU(data_slice.positions[hop].Move(util::DEVICE, util::HOST));
        GUARD_CU(data_slice.total_lengths[hop].SetPointer(h_total_lengths, 1*sizeof (SizeT), util::HOST));
        GUARD_CU(data_slice.total_lengths[hop].Move(util::DEVICE, util::HOST));
        // </TODO>
      } else if (target == util::HOST) {
        // <TODO> extract the results from single CPU, e.g.:
        GUARD_CU(data_slice.neighbors[hop].ForEach(
            h_neighbors,
            [] __host__ __device__(const VertexT &device_val, VertexT &host_val) {
              host_val = device_val;
            },
            h_neighbors_length, util::HOST));
        
          GUARD_CU(data_slice.lengths[hop].ForEach(
              h_lengths,
              [] __host__ __device__(const SizeT &device_val, SizeT &host_val) {
                host_val = device_val;
              },
              nodes, util::HOST));

          GUARD_CU(data_slice.positions[hop].ForEach(
                h_positions,
                [] __host__ __device__(const SizeT &device_val, SizeT &host_val) {
                  host_val = device_val;
                },
                nodes, util::HOST));
          GUARD_CU(data_slice.total_lengths[hop].ForEach(
                  h_total_lengths,
                  [] __host__ __device__(const SizeT &device_val, SizeT &host_val) {
                    host_val = device_val;
                  },
                  1, util::HOST));
        // </TODO>
      }
    } else {  // num_gpus != 1

      // ============ INCOMPLETE TEMPLATE - MULTIGPU ============

      // // TODO: extract the results from multiple GPUs, e.g.:
      // // util::Array1D<SizeT, ValueT *> th_distances;
      // // th_distances.SetName("bfs::Problem::Extract::th_distances");
      // // GUARD_CU(th_distances.Allocate(this->num_gpus, util::HOST));

      // for (int gpu = 0; gpu < this->num_gpus; gpu++)
      // {
      //     auto &data_slice = data_slices[gpu][0];
      //     if (target == util::DEVICE)
      //     {
      //         GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
      //         // GUARD_CU(data_slice.distances.Move(util::DEVICE,
      //         util::HOST));
      //     }
      //     // th_distances[gpu] = data_slice.distances.GetPointer(util::HOST);
      // } //end for(gpu)

      // for (VertexT v = 0; v < nodes; v++)
      // {
      //     int gpu = this -> org_graph -> GpT::partition_table[v];
      //     VertexT v_ = v;
      //     if ((GraphT::FLAG & gunrock::partitioner::Keep_Node_Num) != 0)
      //         v_ = this -> org_graph -> GpT::convertion_table[v];

      //     // h_distances[v] = th_distances[gpu][v_];
      // }

      // // GUARD_CU(th_distances.Release());
    }

    return retval;
  }


  cudaError_t Extract_total_lengths(
    // <TODO> problem specific data to extract
    int hop,
    SizeT *h_total_lengths,
    // </TODO>
    util::Location target = util::DEVICE) {
  cudaError_t retval = cudaSuccess;
  SizeT nodes = this->org_graph->nodes;
  SizeT edges = this->org_graph->edges;

  if (this->num_gpus == 1) {
    auto &data_slice = data_slices[0][0];

    // Set device
    if (target == util::DEVICE) {
      GUARD_CU(util::SetDevice(this->gpu_idx[0]));

      // <TODO> extract the results from single GPU, e.g.:
      GUARD_CU(data_slice.total_lengths[hop].SetPointer(h_total_lengths, 1*sizeof (SizeT), util::HOST));
      GUARD_CU(data_slice.total_lengths[hop].Move(util::DEVICE, util::HOST));
      // </TODO>
    } else if (target == util::HOST) {
      // <TODO> extract the results from single CPU, e.g.:
        GUARD_CU(data_slice.total_lengths[hop].ForEach(
                h_total_lengths,
                [] __host__ __device__(const SizeT &device_val, SizeT &host_val) {
                  host_val = device_val;
                },
                1, util::HOST));
      // </TODO>
    }
  }

  return retval;
}

  /**
   * @brief initialization function.
   * @param     graph       The graph that SSSP processes on
   * @param[in] Location    Memory location to work on
   * \return    cudaError_t Error message(s), if any
   */
  cudaError_t Init(GraphT &graph, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(BaseProblem::Init(graph, target));
    data_slices = new util::Array1D<SizeT, DataSlice>[this->num_gpus];

    // <TODO> get problem specific flags from parameters, e.g.:
    // if (this -> parameters.template Get<bool>("mark-pred"))
    //    this -> flag = this -> flag | Mark_Predecessors;
    // </TODO>
    std::cout << "num_gpus "<< this->num_gpus << std::endl;
    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      data_slices[gpu].SetName("data_slices[" + std::to_string(gpu) + "]");
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));

      GUARD_CU(data_slices[gpu].Allocate(1, target | util::HOST));

      auto &data_slice = data_slices[gpu][0];
      GUARD_CU(data_slice.Init(this->sub_graphs[gpu], this->num_gpus,
                               this->gpu_idx[gpu], target, this->flag));
    }

    return retval;
  }

  /**
   * @brief Reset problem function. Must be called prior to each run.
   * @param[in] src      Source vertex to start.
   * @param[in] location Memory location to work on
   * \return cudaError_t Error message(s), if any
   */
  cudaError_t Reset(
      // <TODO> problem specific data if necessary, eg
      // VertexT src,
      // </TODO>
      util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;

    // Reset data slices
    for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
      GUARD_CU(data_slices[gpu]->Reset(target));
      GUARD_CU(data_slices[gpu].Move(util::HOST, target));
    }

    // <TODO> Additional problem specific initialization
    // </TODO>

    GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
    return retval;
  }
};

}  // namespace hello
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
