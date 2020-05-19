// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * Template_enactor.cuh
 *
 * @brief hello Problem Enactor
 */

#pragma once

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/enactor_iteration.cuh>
#include <gunrock/app/enactor_loop.cuh>
#include <gunrock/oprtr/oprtr.cuh>

// <TODO> change includes
#include <gunrock/app/hello/hello_problem.cuh>
// </TODO>

namespace gunrock {
namespace app {
// <TODO> change namespace
namespace hello {
// </TODO>

/**
 * @brief Speciflying parameters for hello Enactor
 * @param parameters The util::Parameter<...> structure holding all parameter
 * info \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_enactor(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(app::UseParameters_enactor(parameters));

  // <TODO> if needed, add command line parameters used by the enactor here
  // </TODO>

  return retval;
}

/**
 * @brief defination of hello iteration loop
 * @tparam EnactorT Type of enactor
 */
template <typename EnactorT>
struct NeighborsIterationLoop
    : public IterationLoopBase<EnactorT, Use_FullQ | Push
                               // <TODO>if needed, stack more option, e.g.:
                               // | (((EnactorT::Problem::FLAG &
                               // Mark_Predecessors) != 0) ? Update_Predecessors
                               // : 0x0)
                               // </TODO>
                               > {
  typedef typename EnactorT::VertexT VertexT;
  typedef typename EnactorT::SizeT SizeT;
  typedef typename EnactorT::ValueT ValueT;
  typedef typename EnactorT::Problem::GraphT::CsrT CsrT;
  typedef typename EnactorT::Problem::GraphT::GpT GpT;

  typedef IterationLoopBase<EnactorT, Use_FullQ | Push
                            // <TODO> add the same options as in template
                            // parameters here, e.g.: |
                            // (((EnactorT::Problem::FLAG & Mark_Predecessors)
                            // != 0) ? Update_Predecessors : 0x0)
                            // </TODO>
                            >
      BaseIterationLoop;

      NeighborsIterationLoop() : BaseIterationLoop() {}

  /**
   * @brief Core computation of hello, one iteration
   * @param[in] peer_ Which GPU peers to work on, 0 means local
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Core(int peer_ = 0) {
    // --
    // Alias variables

    auto &data_slice = this->enactor->problem->data_slices[this->gpu_num][0];

    auto &enactor_slice =
        this->enactor
            ->enactor_slices[this->gpu_num * this->enactor->num_gpus + peer_];

    auto &enactor_stats = enactor_slice.enactor_stats;
    auto &graph = data_slice.sub_graph[0];
    auto &frontier = enactor_slice.frontier;
    auto &oprtr_parameters = enactor_slice.oprtr_parameters;
    auto &retval = enactor_stats.retval;
    auto &iteration = enactor_stats.iteration;
    std::cout << "=============== iteration " << iteration << " ================ " << std::endl;
    // <TODO> add problem specific data alias here:
    int hop = this->enactor->hop;
    auto &positions = data_slice.positions[hop];
    auto &lengths = data_slice.lengths[hop];
    // </TODO>
    std::cout << "Running for 121221212121211 hop " << hop << std::endl;
  #define S1 25
  #define S2 10
  
    if (hop == 0) {
      auto &neighbors = data_slice.neighbors[hop];
      {
        auto &total_lengths = data_slice.total_lengths[hop];
        util::Array1D<SizeT, VertexT> *null_frontier = NULL;        
        auto advance_op = [
                            neighbors, positions, lengths
        ] __host__ __device__(const VertexT &src, VertexT &dest,
                              const SizeT &edge_id, const VertexT &input_item,
                              const SizeT &input_pos, SizeT &output_pos) -> bool {
          atomicAdd (&lengths[src], 1);
          return false;
        };

        
        GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V>(
            graph.csr(), null_frontier, null_frontier, oprtr_parameters,
            advance_op));

        cudaDeviceSynchronize ();

        positions.ForAll (
          [total_lengths, lengths] __device__ __host__ (SizeT* positions, VertexT i) {
            positions[i] = atomicAdd (&total_lengths[0], lengths[i]);
          }, data_slice.sub_graph[0].nodes, util::DEVICE, data_slice.stream);

        cudaDeviceSynchronize ();
      }

      {
        util::Array1D<SizeT, VertexT> *null_frontier = NULL;

        GUARD_CU (lengths.ForEach (
          [] __device__ __host__ (SizeT &x) {x = 0;}, graph.nodes, util::DEVICE, data_slice.stream));

        cudaDeviceSynchronize ();

        // advance operation
        auto advance_op = [
                              neighbors, positions, lengths
        ] __host__ __device__(const VertexT &src, VertexT &dest,
                              const SizeT &edge_id, const VertexT &input_item,
                              const SizeT &input_pos, SizeT &output_pos) -> bool {
          if (edge_id < S1) {
            auto l = atomicAdd (&lengths[src], 1);
            neighbors[src*S1 + edge_id] = dest;
          }
          return false;
        };

        GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V>(
            graph.csr(), null_frontier, null_frontier, oprtr_parameters,
            advance_op));

        cudaDeviceSynchronize ();
      }
    } else {
      auto &prev_neighbors = data_slice.neighbors[hop-1];
      auto &prev_positions = data_slice.positions[hop-1];
      auto &prev_lengths = data_slice.lengths[hop-1];

      auto &total_lengths = data_slice.total_lengths[hop];

      {        
        util::Array1D<SizeT, VertexT> *null_frontier = NULL;        
        auto advance_op = [
                            prev_lengths, lengths
        ] __host__ __device__(const VertexT &src, VertexT &dest,
                              const SizeT &edge_id, const VertexT &input_item,
                              const SizeT &input_pos, SizeT &output_pos) -> bool {
          lengths[src] = S2;
          return false;
        };

        GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V>(
            graph.csr(), null_frontier, null_frontier, oprtr_parameters,
            advance_op));

        cudaDeviceSynchronize ();
      }

      util::CpuTimer cpu_timer;
      cpu_timer.Start();
      std::vector<VertexT> h_prev_neighbors = std::vector<VertexT>(graph.nodes*S1);
      GUARD_CU(prev_neighbors.SetPointer(h_prev_neighbors.data(), graph.nodes*S1*sizeof(VertexT), util::HOST));
      GUARD_CU(prev_neighbors.Move(util::DEVICE, util::HOST));
      std::vector<std::vector<VertexT>> src_to_roots = std::vector<std::vector<VertexT>>(graph.nodes);
      for (VertexT v = 0; v < graph.nodes; v++) {
        src_to_roots[h_prev_neighbors[v]].push_back(v);
      }

      std::vector<VertexT> h_csr_transpose_roots = std::vector<VertexT>(graph.nodes*S1);
      std::vector<VertexT> h_csr_src_pos = std::vector<VertexT>(2*graph.nodes);
      SizeT iter = 0;
      for (VertexT s = 0; s < graph.nodes; s++) {
        h_csr_src_pos[2*s] = iter;
        for (auto r : src_to_roots[s]) {
          h_csr_transpose_roots[iter] = r;
          iter++;
        }
        h_csr_src_pos[2*s+1] = iter;
      }

      data_slice.InitNeighborsForHop (graph.nodes*S1*S2, hop, util::DEVICE);
      VertexT* d_csr_src_pos;
      VertexT* d_csr_transpose_roots;

      GUARD_CU(cudaMalloc(&d_csr_src_pos, h_csr_src_pos.size()*sizeof(VertexT)));
      GUARD_CU(cudaMalloc(&d_csr_transpose_roots, h_csr_transpose_roots.size()*sizeof(VertexT)));
      GUARD_CU(cudaMemcpy(d_csr_src_pos, h_csr_src_pos.data(), 
                          h_csr_src_pos.size()*sizeof(VertexT), cudaMemcpyHostToDevice));
      GUARD_CU(cudaMemcpy(d_csr_transpose_roots, h_csr_transpose_roots.data(), 
                          h_csr_transpose_roots.size()*sizeof(VertexT), cudaMemcpyHostToDevice));
      cpu_timer.Stop();
      this->enactor->exclude_time += cpu_timer.ElapsedMillis();

      auto &neighbors = data_slice.neighbors[hop];

      {
        util::Array1D<SizeT, VertexT> *null_frontier = NULL;

        GUARD_CU (lengths.ForEach (
          [] __device__ __host__ (SizeT &x) {x = 0;}, graph.nodes, util::DEVICE, data_slice.stream));

        GUARD_CU(cudaDeviceSynchronize ());

        // advance operation
        auto advance_op = [
                              neighbors, positions, lengths, prev_positions, prev_lengths, prev_neighbors,
                              d_csr_src_pos, d_csr_transpose_roots
        ] __host__ __device__(const VertexT &src, VertexT &dest,
                              const SizeT &edge_id, const VertexT &input_item,
                              const SizeT &input_pos, SizeT &output_pos) -> bool {
          for (SizeT q = d_csr_src_pos[2*src]; q < d_csr_src_pos[2*src] + d_csr_src_pos[2*src+1]; q++) {
            VertexT root = d_csr_transpose_roots[q];

            if (edge_id < S2) {
              auto l = atomicAdd (&lengths[root], 1);
              neighbors[root*S2*S1+l] = dest;
            }
          }
          return false;
        };

        GUARD_CU(oprtr::Advance<oprtr::OprtrType_V2V>(
            graph.csr(), null_frontier, null_frontier, oprtr_parameters,
            advance_op));

        GUARD_CU(cudaDeviceSynchronize ());
      }

      GUARD_CU(cudaFree(d_csr_src_pos));
      GUARD_CU(cudaFree(d_csr_transpose_roots));
    }

    // Get back the resulted frontier length
    //GUARD_CU(frontier.work_progress.GetQueueLength(
    //    frontier.queue_index, frontier.queue_length, false,
    //    oprtr_parameters.stream, true));
    std::cout << __LINE__ << ": frontier queue length " << frontier.queue_length << std::endl;

    // </TODO>

    return retval;
  }

  bool Stop_Condition(int gpu_num = 0) {
    auto &data_slice = this->enactor->problem->data_slices[this->gpu_num][0];
    auto &enactor_slices = this->enactor->enactor_slices;
    auto iter = enactor_slices[0].enactor_stats.iteration;

    // user defined stop condition
    if (iter == 1) return true;
    return false;
  }

  /**
   * @brief Routine to combine received data and local data
   * @tparam NUM_VERTEX_ASSOCIATES Number of data associated with each
   * transmition item, typed VertexT
   * @tparam NUM_VALUE__ASSOCIATES Number of data associated with each
   * transmition item, typed ValueT
   * @param  received_length The numver of transmition items received
   * @param[in] peer_ which peer GPU the data came from
   * \return cudaError_t error message(s), if any
   */
  template <int NUM_VERTEX_ASSOCIATES, int NUM_VALUE__ASSOCIATES>
  cudaError_t ExpandIncoming(SizeT &received_length, int peer_) {
    // ================ INCOMPLETE TEMPLATE - MULTIGPU ====================
    assert (false);
    auto &data_slice = this->enactor->problem->data_slices[this->gpu_num][0];
    auto &enactor_slice =
        this->enactor
            ->enactor_slices[this->gpu_num * this->enactor->num_gpus + peer_];
    // auto iteration = enactor_slice.enactor_stats.iteration;
    // TODO: add problem specific data alias here, e.g.:
    // auto         &distances          =   data_slice.distances;

    auto expand_op = [
                         // TODO: pass data used by the lambda, e.g.:
                         // distances
    ] __host__ __device__(VertexT & key, const SizeT &in_pos,
                          VertexT *vertex_associate_ins,
                          ValueT *value__associate_ins) -> bool {
      // TODO: fill in the lambda to combine received and local data, e.g.:
      // ValueT in_val  = value__associate_ins[in_pos];
      // ValueT old_val = atomicMin(distances + key, in_val);
      // if (old_val <= in_val)
      //     return false;
      return true;
    };

    cudaError_t retval =
        BaseIterationLoop::template ExpandIncomingBase<NUM_VERTEX_ASSOCIATES,
                                                       NUM_VALUE__ASSOCIATES>(
            received_length, peer_, expand_op);
    return retval;
  }
};  // end of helloIteration

/**
 * @brief Template enactor class.
 * @tparam _Problem Problem type we process on
 * @tparam ARRAY_FLAG Flags for util::Array1D used in the enactor
 * @tparam cudaHostRegisterFlag Flags for util::Array1D used in the enactor
 */
template <typename _Problem, util::ArrayFlag ARRAY_FLAG = util::ARRAY_NONE,
          unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault>
class Enactor
    : public EnactorBase<
          typename _Problem::GraphT, typename _Problem::GraphT::VertexT,
          typename _Problem::GraphT::ValueT, ARRAY_FLAG, cudaHostRegisterFlag> {
 public:
  typedef _Problem Problem;
  typedef typename Problem::SizeT SizeT;
  typedef typename Problem::VertexT VertexT;
  typedef typename Problem::GraphT GraphT;
  typedef typename GraphT::VertexT LabelT;
  typedef typename GraphT::ValueT ValueT;
  typedef EnactorBase<GraphT, LabelT, ValueT, ARRAY_FLAG, cudaHostRegisterFlag>
      BaseEnactor;
  typedef Enactor<Problem, ARRAY_FLAG, cudaHostRegisterFlag> EnactorT;
  typedef NeighborsIterationLoop<EnactorT> NeighborsIterationT;
  Problem *problem;
  NeighborsIterationT *neighborsIterations;
  int hop;
  double exclude_time;
  /**
   * @brief hello constructor
   */
  Enactor() : BaseEnactor("Template"), problem(NULL), exclude_time(0.0) {
    // <TODO> change according to algorithmic needs
    this->max_num_vertex_associates = 0;
    this->max_num_value__associates = 1;
    // </TODO>
  }

  /**
   * @brief hello destructor
   */
  virtual ~Enactor() { /*Release();*/
  }

  /*
   * @brief Releasing allocated memory space
   * @param target The location to release memory from
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Release(util::Location target = util::LOCATION_ALL) {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(BaseEnactor::Release(target));
    delete[] neighborsIterations;
    neighborsIterations = NULL;
    problem = NULL;
    return retval;
  }

  /**
   * @brief Initialize the problem.
   * @param[in] problem The problem object.
   * @param[in] target Target location of data
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Init(Problem &problem, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    this->problem = &problem;

    // Lazy initialization
    GUARD_CU(BaseEnactor::Init(
        problem, Enactor_None,
        // <TODO> change to how many frontier queues, and their types
        2, NULL,
        // </TODO>
        target, false));
    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
      auto &enactor_slice = this->enactor_slices[gpu * this->num_gpus + 0];
      auto &graph = problem.sub_graphs[gpu];
      GUARD_CU(enactor_slice.frontier.Allocate(graph.nodes, graph.edges,
                                               this->queue_factors));
    }

    assert (this->num_gpus == 1);
    neighborsIterations = new NeighborsIterationT[this->num_gpus];
    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      GUARD_CU(neighborsIterations[gpu].Init(this, gpu));
    }

    GUARD_CU(this->Init_Threads(
        this, (CUT_THREADROUTINE) & (GunrockThread<EnactorT>)));
    return retval;
  }

  /**
   * @brief one run of hello, to be called within GunrockThread
   * @param thread_data Data for the CPU thread
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Run(ThreadSlice &thread_data) {
    // gunrock::app::Iteration_Loop<
    //     // <TODO> change to how many {VertexT, ValueT} data need to communicate
    //     //       per element in the inter-GPU sub-frontiers
    //     0, 0,
    //     // </TODO>
    //     VertexOffsetsIterationT>(thread_data, vertexOffsetsIterations[thread_data.thread_num]);

    gunrock::app::Iteration_Loop<
        // <TODO> change to how many {VertexT, ValueT} data need to communicate
        //       per element in the inter-GPU sub-frontiers
        0, 0,
        // </TODO>
        NeighborsIterationT>(thread_data, neighborsIterations[thread_data.thread_num]);
    return cudaSuccess;
  }

  /**
   * @brief Reset enactor
...
   * @param[in] target Target location of data
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Reset(
      // <TODO> problem specific data if necessary, eg
      int _hop = 0,
      // </TODO>
      util::Location target = util::DEVICE) {
    typedef typename GraphT::GpT GpT;
    cudaError_t retval = cudaSuccess;
    GUARD_CU(BaseEnactor::Reset(target));
    this->hop = _hop;
    // <TODO> Initialize frontiers according to the algorithm:
    // In this case, we add a single `src` to the frontier
    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      if (this->num_gpus == 1) {
        this->thread_slices[gpu].init_size = 1;
        for (int peer_ = 0; peer_ < this->num_gpus; peer_++) {
          auto &frontier =
              this->enactor_slices[gpu * this->num_gpus + peer_].frontier;
          auto &data_slice = problem->data_slices[gpu * this->num_gpus + peer_][0];
          frontier.queue_length = (peer_ == 0) ? data_slice.sub_graph[0].nodes : 0;
          std::cout <<" frontier.queue_length " << frontier.queue_length << std::endl;
          if (peer_ == 0) {
            GUARD_CU(frontier.V_Q()->ForAll(
                [] __host__ __device__(VertexT * v, VertexT i) { *(v + i) = i; }, frontier.queue_length, target,
                0));
          }
        }
      } else {
        this->thread_slices[gpu].init_size = 0;
        for (int peer_ = 0; peer_ < this->num_gpus; peer_++) {
          this->enactor_slices[gpu * this->num_gpus + peer_]
              .frontier.queue_length = 0;
        }
      }
    }
    // </TODO>

    GUARD_CU(BaseEnactor::Sync());
    return retval;
  }

  /**
   * @brief Enacts a hello computing on the specified graph.
...
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Enact(
      // <TODO> problem specific data if necessary, eg
      int _hop
      // </TODO>
  ) {
    cudaError_t retval = cudaSuccess;
    this->hop = _hop;
    this->enactor_slices[0].enactor_stats.iteration = 0;
    GUARD_CU(this->Run_Threads(this));
    util::PrintMsg("GPU Template Done.", this->flag & Debug);
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
