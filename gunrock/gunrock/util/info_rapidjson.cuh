// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * info_rapidjson.cuh
 *
 * @brief Running statistic collector with rapidjson
 */

#pragma once

#include <cstdio>
#include <vector>
#include <ctime>
#include <time.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/filewritestream.h>
#include <gunrock/util/gitsha1.h>
#include <gunrock/util/sysinfo_rapidjson.h>

/* this is the "stringize macro macro" hack */
#define STR(x) #x
#define XSTR(x) STR(x)

namespace gunrock {
namespace util {

template <typename ParametersT>
cudaError_t UseParameters_info(ParametersT &parameters) {
  cudaError_t retval = cudaSuccess;
  
  GUARD_CU(parameters.template Use<bool>(
      "json",
      util::OPTIONAL_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      false, "Whether to output statistics in json format", __FILE__,
      __LINE__));

  GUARD_CU(parameters.template Use<std::string>(
      "jsonfile",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      "", "Filename to output statistics in json format", __FILE__, __LINE__));

  GUARD_CU(parameters.template Use<std::string>(
      "jsondir",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      "", "Directory to output statistics in json format", __FILE__, __LINE__));

  GUARD_CU(parameters.template Use<std::string>(
      "tag",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      "", "Tag to better describe and identify json outputs", __FILE__,
      __LINE__));

  // GUARD_CU(parameters.template Use<uint64_t>(
  //   "filtered-srcs",
  //   util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::INTERNAL_PARAMETER, 0,
  //   "Array of filtered source vertices", __FILE__, __LINE__));

  return retval;
}

/**
 * @brief Info data structure contains running statistics.
 */
struct Info {
 private:
  double total_elapsed;               // sum of running times
  double elapsed;                     // average of running times
  double max_elapsed;                 // maximum running time
  double min_elapsed;                 // minimum running time
  double stddev_process_time;         // std. deviation of running times
  std::vector<double> process_times;  // array of running times (raw)
  std::vector<double> _process_times; // array of running times (filtered)
  int num_runs;                       // number of runs
  int64_t nodes_visited;
  int64_t edges_visited;
  double max_m_teps;                 // maximum MTEPS
  double min_m_teps;                 // minimum MTEPS
  double m_teps;
  int64_t search_depth;
  double avg_duty;
  int64_t nodes_queued;
  int64_t edges_queued;
  double nodes_redundance;
  double edges_redundance;
  // double     load_time;
  double preprocess_time;
  double postprocess_time;
  double write_time;
  double total_time;
  std::string time_str;
  std::string algorithm_name;

  util::Parameters *parameters;
  std::string json_filename;
  std::FILE *json_file;
  char *json_buffer;
  size_t buff_size;
  rapidjson::FileWriteStream *json_stream;
  rapidjson::PrettyWriter<rapidjson::FileWriteStream> *json_writer;

 public:
  /**
   * @brief Info default constructor
   */
  Info() { AssignInitValues(); }

  template <typename GraphT>
  Info(std::string algorithm_name, util::Parameters &parameters,
       GraphT &graph) {
    AssignInitValues();
    Init(algorithm_name, parameters, graph);
  }

  void AssignInitValues() {
    parameters = NULL;
    json_filename = "";
    json_buffer = NULL;
    buff_size = 0;
    json_stream = NULL;
    json_writer = NULL;
    json_file = NULL;
  }

  ~Info() { Release(); }

  cudaError_t Release() {
    parameters = NULL;
    json_filename = "";
    delete[] json_buffer;
    json_buffer = NULL;
    delete json_stream;
    json_stream = NULL;
    delete json_writer;
    json_writer = NULL;
    return cudaSuccess;
  }

  /**
   * @brief Initialization process for Info.
   *
   * @param[in] algorithm_name Algorithm name.
   * @param[in] args Command line arguments.
   */
  void InitBase(std::string algorithm_name, util::Parameters &parameters) {

    std::transform (algorithm_name.begin(), algorithm_name.end(), 
                    algorithm_name.begin(), ::tolower);

    this->algorithm_name = algorithm_name;
    this->parameters = &parameters;

    total_elapsed = 0;
    max_elapsed = 0;
    min_elapsed = 1e26;
    num_runs = 0;

    time_t now = time(NULL);

    long        ms; // Milliseconds
    time_t          s;  // Seconds
    struct timespec spec;

    clock_gettime(CLOCK_REALTIME, &spec);

    s  = spec.tv_sec;
    ms = round(spec.tv_nsec / 1.0e6); // Convert nanoseconds to milliseconds
    if (ms > 999) {
        s++;
        ms = 0;
    }

    std::string time_s = std::string(ctime(&now));
    std::string time_ms = std::to_string(ms);

    time_str = time_s;
    std::string time_str_filename = time_s.substr(0, time_s.size() - 5) + time_ms + ' ' + time_s.substr(time_s.length() - 5);

    if (parameters.Get<bool>("json")) {
      json_filename = "";
    } else if (parameters.Get<std::string>("jsonfile") != "") {
      json_filename = parameters.Get<std::string>("jsonfile");
    } else if (parameters.Get<std::string>("jsondir") != "") {
      std::string dataset = parameters.Get<std::string>("dataset");
      std::string dir = parameters.Get<std::string>("jsondir");
      json_filename = dir + "/" + algorithm_name + "_" +
                      ((dataset != "") ? (dataset + "_") : "") + time_str_filename +
                      ".json";

      char bad_chars[] = ":\n";
      for (unsigned int i = 0; i < strlen(bad_chars); ++i) {
        json_filename.erase(std::remove(json_filename.begin(),
                                        json_filename.end(), bad_chars[i]),
                            		json_filename.end());
      }
    } else {
      return;
    }

    buff_size = 65536;
    if (json_buffer == NULL) json_buffer = new char[buff_size];
    if (json_filename != "") {
      json_file = std::fopen(json_filename.c_str(), "w");
    } else {
      json_file = stdout;
    }
    json_stream =
        new rapidjson::FileWriteStream(json_file, json_buffer, buff_size);
    json_writer =
        new rapidjson::PrettyWriter<rapidjson::FileWriteStream>(*json_stream);
    json_writer->StartObject();

    SetVal("engine", "Gunrock");
    SetVal("command-line", parameters.Get_CommandLine());
    
    // Update this date when JSON Schema is changed:
    SetVal("json-schema", "2019-09-20");  
    
    // SetVal("sysinfo", sysinfo.getSysinfo());
    // SetVal("gpuinfo", gpuinfo.getGpuinfo());
    // SetVal("userinfo", userinfo.getUserinfo());

#ifdef BOOST_FOUND
#if BOOST_COMP_CLANG
    SetVal("compiler", BOOST_COMP_CLANG_NAME);
    SetVal("compiler-version", BOOST_COMP_CLANG_DETECTION);
#elif BOOST_COMP_GNUC
    SetVal("compiler", BOOST_COMP_GNUC_NAME);
    SetVal("compiler-version", BOOST_COMP_GNUC_DETECTION);
#endif
#else
#ifdef __clang__
    SetVal("compiler", "Clang");
    SetVal("compiler-version", (__clang_major__ % 100) * 10000000 +
                                   (__clang_minor__ % 100) * 100000 +
                                   (__clang_patchlevel__ % 100000));
#else
    SetVal("compiler", "Gnu GCC C/C++");
#ifdef __GNUC_PATCHLEVEL__
    SetVal("compiler-version", (__GNUC__ % 100) * 10000000 +
                                   (__GNUC_MINOR__ % 100) * 100000 +
                                   (__GNUC_PATCHLEVEL__ % 100000));
#else
    SetVal("compiler-version",
           (__GNUC__ % 100) * 10000000 + (__GNUC_MINOR__ % 100) * 100000);
#endif
#endif
#endif

    SetVal("time", time_str);
    SetVal("gunrock-version", XSTR(GUNROCKVERSION));
    SetVal("git-commit-sha", g_GIT_SHA1);
    SetVal("load-time", parameters.Get<float>("load-time"));
    SetVal("primitive", algorithm_name);
  }

  /**
   * @brief Initialization process for Info.
   * @param[in] algorithm_name Algorithm name.
   * @param[in] parameters running parameters.
   * @param[in] graph The graph.
   */
  template <typename GraphT>
  void Init(std::string algorithm_name, util::Parameters &parameters,
            GraphT &graph) {
    InitBase(algorithm_name, parameters);
    // if not set or something is wrong, set it to the largest vertex ID
    // if (info["destination_vertex"].get_int64() < 0 ||
    //    info["destination_vertex"].get_int64() >= graph.nodes)
    //    info["destination_vertex"] = graph.nodes - 1;

    SetVal("stddev-degree", graph::GetStddevDegree(graph));
    SetVal("num-vertices", graph.nodes);
    SetVal("num-edges", graph.edges);
  }

  template <typename DummyT, typename T>
  void SetBool(DummyT name, const T &val) {
    // util::PrintMsg("Escape val, name = " + name);
    if (json_writer != NULL) json_writer->Null();
  }

  template <typename DummyT>
  void SetBool(DummyT name, const bool &val) {
    if (json_writer != NULL) json_writer->Bool(val);
  }

  template <typename DummyT, typename T>
  void SetInt(DummyT name, const T &val) {
    // util::PrintMsg("Escape val, name = " + name);
    if (json_writer != NULL) json_writer->Null();
  }

  template <typename DummyT>
  void SetInt(DummyT name, const int &val) {
    if (json_writer != NULL) json_writer->Int(val);
  }

  template <typename DummyT, typename T>
  void SetUint(DummyT name, const T &val) {
    // util::PrintMsg("Escape val, name = " + name);
    if (json_writer != NULL) json_writer->Null();
  }

  template <typename DummyT>
  void SetUint(DummyT name, const unsigned int &val) {
    if (json_writer != NULL) json_writer->Uint(val);
  }

  template <typename DummyT, typename T>
  void SetInt64(DummyT name, const T &val) {
    // util::PrintMsg("Escape val, name = " + name);
    if (json_writer != NULL) json_writer->Null();
  }

  template <typename DummyT>
  void SetInt64(DummyT name, const int64_t &val) {
    if (json_writer != NULL) json_writer->Int64(val);
  }

  template <typename DummyT, typename T>
  void SetUint64(DummyT name, const T &val) {
    // util::PrintMsg("Escape val, name = " + name);
    if (json_writer != NULL) json_writer->Null();
  }

  template <typename DummyT>
  void SetUint64(DummyT name, const uint64_t &val) {
    if (json_writer != NULL) json_writer->Uint64(val);
  }

  template <typename DummyT, typename T>
  void SetDouble(DummyT name, const T &val) {
    // util::PrintMsg("Escape val, name = " + name);
    if (json_writer != NULL) json_writer->Null();
  }

  template <typename DummyT>
  void SetDouble(DummyT name, const float &val) {
    if (json_writer != NULL) json_writer->Double(double(val));
  }

  template <typename DummyT>
  void SetDouble(DummyT name, const double &val) {
    if (json_writer != NULL) json_writer->Double(val);
  }

  template <typename T>
  void SetVal(std::string name, const T &val, bool write_name = true) {
    if (json_writer == NULL) return;
    if (write_name) json_writer->Key(name.c_str());
    auto tidx = std::type_index(typeid(T));

    if (tidx == std::type_index(typeid(bool)))
      SetBool(name, val);
    else if (tidx == std::type_index(typeid(char)) ||
             tidx == std::type_index(typeid(signed char)) ||
             tidx == std::type_index(typeid(short)) ||
             tidx == std::type_index(typeid(int)))
      SetInt(name, val);
    else if (tidx == std::type_index(typeid(unsigned char)) ||
             tidx == std::type_index(typeid(unsigned short)) ||
             tidx == std::type_index(typeid(unsigned int)))
      SetUint(name, val);
    else if (tidx == std::type_index(typeid(long)) ||
             tidx == std::type_index(typeid(long long)))
      SetInt64(name, val);
    else if (tidx == std::type_index(typeid(unsigned long)) ||
             tidx == std::type_index(typeid(unsigned long long)))
      SetUint64(name, val);
    else if (tidx == std::type_index(typeid(float)) ||
             tidx == std::type_index(typeid(double)) ||
             tidx == std::type_index(typeid(long double)))
      SetDouble(name, val);
    else {
      std::ostringstream ostr;
      ostr << val;
      std::string str = ostr.str();
      json_writer->String(str.c_str());
    }
  }

  template <typename T>
  void SetVal(std::string name, const std::vector<T> &vec) {
    if (json_writer == NULL) return;
    // TODO: update parameters to support "ALWAYS_ARRAY" type
    // currently using a hack to make sure tag is always an 
    // array in JSON. This is also required for fields such
    // as srcs, process-times, etc.
    if (vec.size() == 1 && (name.compare("tag") != 0)) {
      SetVal(name, vec.front());
    } else {
      json_writer->Key(name.c_str());
      json_writer->StartArray();
      for (auto it = vec.begin(); it != vec.end(); it++)
        SetVal(name, *it, false);
      json_writer->EndArray();
    }
  }

  template <typename T>
  void SetVal(std::string name, const std::vector<std::pair<T, T>> &vec) {
    if (json_writer == NULL) return;
    json_writer->Key(name.c_str());
    json_writer->StartObject();
    for (auto it = vec.begin(); it != vec.end(); it++) {
      SetVal(it->first.c_str(), it->second);
    }
    json_writer->EndObject();
  }

  void CollectSingleRun(double single_elapsed) {
    total_elapsed += single_elapsed;
    process_times.push_back(single_elapsed);
    num_runs++;
  }

  /**
   * @brief Compute statistics common to all primitives.
   *
   * @param[in] enactor_stats
   * @param[in] elapsed
   * @param[in] labels
   * @param[in] get_traversal_stats
   */
  template <typename EnactorT, typename T>
  cudaError_t ComputeCommonStats(EnactorT &enactor, const T *labels = NULL,
                                 bool get_traversal_stats = false) {
    cudaError_t retval = cudaSuccess;
    double total_lifetimes = 0;
    double total_runtimes = 0;

    // traversal stats
    edges_queued = 0;
    nodes_queued = 0;
    search_depth = 0;
    nodes_visited = 0;
    edges_visited = 0;
    m_teps = 0.0f;
    edges_redundance = 0.0f;
    nodes_redundance = 0.0f;

    std::vector<int> device_list = parameters->Get<std::vector<int>>("device");
    int num_gpus = device_list.size();
    auto graph = enactor.problem->org_graph[0];

    for (int gpu = 0; gpu < num_gpus; ++gpu) {
      int my_gpu_idx = device_list[gpu];
      if (num_gpus != 1) {
        GUARD_CU(util::SetDevice(my_gpu_idx));
      }
      GUARD_CU(cudaDeviceSynchronize());

      for (int peer = 0; peer < num_gpus; ++peer) {
        auto &estats =
            enactor.enactor_slices[gpu * num_gpus + peer].enactor_stats;
        if (get_traversal_stats) {
          edges_queued += estats.edges_queued[0];
          GUARD_CU(estats.edges_queued.Move(util::DEVICE, util::HOST));
          edges_queued += estats.edges_queued[0];

          nodes_queued += estats.nodes_queued[0];
          GUARD_CU(estats.nodes_queued.Move(util::DEVICE, util::HOST));
          nodes_queued += estats.nodes_queued[0];

          if (estats.iteration > search_depth) {
            search_depth = estats.iteration;
          }
        }
        total_lifetimes += estats.total_lifetimes;
        total_runtimes += estats.total_runtimes;
      }
    }

#ifdef RECORD_PER_ITERATION_STATS
    if (get_traversal_stats) {
      // TODO: collect info for multi-GPUs
      EnactorStats *estats = enactor_stats;
      json_spirit::mArray per_iteration_advance_runtime;
      json_spirit::mArray per_iteration_advance_mteps;
      json_spirit::mArray per_iteration_advance_input_frontier;
      json_spirit::mArray per_iteration_advance_output_frontier;
      json_spirit::mArray per_iteration_advance_dir;
      GetPerIterationAdvanceStats(
          estats.per_iteration_advance_time, estats.per_iteration_advance_mteps,
          estats.per_iteration_advance_input_edges,
          estats.per_iteration_advance_output_edges,
          estats.per_iteration_advance_direction, per_iteration_advance_runtime,
          per_iteration_advance_mteps, per_iteration_advance_input_frontier,
          per_iteration_advance_output_frontier, per_iteration_advance_dir);

      SetVal("per_iteration_advance_runtime", per_iteration_advance_runtime);
      SetVal("per_iteration_advance_mteps", per_iteration_advance_mteps);
      SetVal("per_iteration_advance_input_frontier",
             per_iteration_advance_input_frontier);
      SetVal("per_iteration_advance_output_frontier",
             per_iteration_advance_output_frontier);
      SetVal("per_iteration_advance_direction", per_iteration_advance_dir);
    }
#endif

    avg_duty = (total_lifetimes > 0)
                   ? double(total_runtimes) / total_lifetimes * 100.0
                   : 0.0f;
   
    // Throw out results that are 2 standard deviations away from
    // the mean (processing times) and recompute the average.
    // filtering: start
    elapsed = total_elapsed / num_runs;

    if (num_runs > 1) {
      double variance = 0.0;

      for (auto i = process_times.begin(); i != process_times.end(); i++)
        variance += pow(*i - elapsed, 2);
      
      variance = variance / num_runs;
      stddev_process_time = sqrt(variance);

      auto lower_limit = elapsed - (2*stddev_process_time);
      auto upper_limit = elapsed + (2*stddev_process_time);

      // TODO: Check if this works with cases where we don't have
      // multiple srcs, instead all process times maybe use one src
      // (for example, src = 0 or largestdegree, etc.)
      std::vector<int64_t> srcs;
      if (parameters->Have("srcs")) {
        srcs = parameters->Get<std::vector<int64_t>>("srcs");
      }

      std::vector<std::pair<int64_t, double>> delete_runs;

      for(auto i = 0; i < process_times.size(); i++) {
          delete_runs.push_back(std::make_pair((int64_t)srcs[i], (double)process_times[i]));
      }

      // for (auto q = delete_runs.begin(); q != delete_runs.end(); ++q) 
      //     std::cout << ' ' << (*q).first << ' ' << (*q).second << std::endl;

      delete_runs.erase(std::remove_if(
                          delete_runs.begin(), delete_runs.end(),
                          [lower_limit, upper_limit](const std::pair<const int64_t, 
                                                                    const double>& x) {
                            return ((x.second < lower_limit) || (x.second > upper_limit));
                          }), delete_runs.end());

      std::vector<int64_t> _srcs;

      for (auto q = delete_runs.begin(); q != delete_runs.end(); ++q) {
        _process_times.push_back((double)(*q).second);
        _srcs.push_back((int64_t)(*q).first);
      }

      // filtering: end
      
      total_elapsed = 0.0;
      for (auto i = _process_times.begin(); i != _process_times.end(); i++) {
        total_elapsed += *i;
      }
      
      elapsed = total_elapsed / _process_times.size();
      SetVal("filtered-srcs", _srcs);
    }
    
    SetVal("average-duty", avg_duty);
    SetVal("search-depth", search_depth);

    if (get_traversal_stats) {
      SetVal("edges-queued", edges_queued);
      SetVal("nodes-queued", nodes_queued);
    }

    // TODO: compute traversal stats
    if (get_traversal_stats) {
      if (labels != NULL)
        for (int64_t v = 0; v < graph.nodes; ++v) {
          if (util::isValid(labels[v]) &&
              labels[v] != util::PreDefinedValues<T>::MaxValue) {
            ++nodes_visited;
            edges_visited += graph.GetNeighborListLength(v);
          }
        }
      if (algorithm_name == "BC") {
        // for betweenness should count the backward phase too.
        edges_visited = 2 * edges_queued;
      } else if (algorithm_name == "PR") {
        edges_visited = graph.edges;
        nodes_visited = graph.nodes;
      }

      if (nodes_queued >
          nodes_visited) {  // measure duplicate nodes put through queue
        nodes_redundance =
            ((double)nodes_queued - nodes_visited) / nodes_visited;
      }

      if (edges_queued > edges_visited) {
        // measure duplicate edges put through queue
        edges_redundance =
            ((double)edges_queued - edges_visited) / edges_visited;
      }
      nodes_redundance *= 100;
      edges_redundance *= 100;

      m_teps = (double)edges_visited / (elapsed * 1000.0);

      SetVal("nodes-visited", nodes_visited);
      SetVal("edges-visited", edges_visited);
      SetVal("nodes-redundance", nodes_redundance);
      SetVal("edges-redundance", edges_redundance);
    }

    return retval;
  }

  /**
   * @brief Compute statistics common to all traversal primitives.
   * @param[in] enactor The Enactor
   * @param[in] labels
   */
  template <typename EnactorT, typename T>
  void ComputeTraversalStats(EnactorT &enactor, const T *labels = NULL) {
    ComputeCommonStats(enactor, labels, true);
  }

  /**
   * @brief Display running statistics.
   * @param[in] verbose Whether or not to print extra information.
   */
  void DisplayStats(bool verbose = true) {
    int num_runs = parameters->Get<int>("num-runs");
    double elapsed = total_elapsed / num_runs;
    int num_srcs = 0;
    std::vector<int64_t> srcs;
    if (parameters->Have("srcs")) {
      srcs = parameters->Get<std::vector<int64_t>>("srcs");
      num_srcs = srcs.size();
    }

    util::PrintMsg("[" + algorithm_name + "] finished.");
    util::PrintMsg(" avg. elapsed: " + std::to_string(elapsed) + " ms");
    util::PrintMsg(" iterations: " + std::to_string(search_depth));

    if (!verbose) return;
    if (nodes_visited != 0 && nodes_visited < 5) {
      util::PrintMsg("Fewer than 5 vertices visited.");
      return;
    }

    util::PrintMsg(" min. elapsed: " + std::to_string(min_elapsed) + " ms",
                   min_elapsed > 0);
    util::PrintMsg(" max. elapsed: " + std::to_string(max_elapsed) + " ms",
                   max_elapsed > 0);

    util::PrintMsg(" rate: " + std::to_string(m_teps) + " MiEdges/s",
                   m_teps > 0.01);

    util::PrintMsg(" average CTA duty: " + std::to_string(avg_duty) + "%%",
                   avg_duty > 0.01);

    if (nodes_visited != 0 && edges_visited != 0) {
      util::PrintMsg(" src: " + std::to_string(srcs[num_runs % num_srcs]),
                     num_srcs != 0);
      util::PrintMsg(" nodes_visited: " + std::to_string(nodes_visited));
      util::PrintMsg(" edges_visited: " + std::to_string(edges_visited));
    }
    util::PrintMsg(" nodes queued: " + std::to_string(nodes_queued),
                   nodes_queued > 0);
    util::PrintMsg(" edges queued: " + std::to_string(edges_queued),
                   edges_queued > 0);

    util::PrintMsg(" nodes redundance: " + std::to_string(nodes_redundance),
                   nodes_redundance > 0.01);
    util::PrintMsg(" edges redundance: " + std::to_string(edges_redundance),
                   edges_redundance > 0.01);

    util::PrintMsg(" load time: " + parameters->Get<std::string>("load-time") +
                   " ms");
    util::PrintMsg(" preprocess time: " + std::to_string(preprocess_time) +
                   " ms");
    util::PrintMsg(" postprocess time: " + std::to_string(postprocess_time) +
                   " ms");
    // if (parameters -> Get<std::string>("output_filename") != "")
    //    util::PrintMsg(" write time: " + std::to_string(write_time) + " ms");
    util::PrintMsg(" total time: " + std::to_string(total_time) + " ms");

    // TODO: Update DisplayStats to display the new fields and more detailed
    // information about the runs. 
  }

  /* work around to supporting different pair types */
  void getGpuinfo() {
    cudaDeviceProp devProps;
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) /* no valid devices */
    {
      return;
    }
    int dev = 0;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&devProps, dev);
    if (json_writer == NULL) return;
    json_writer->Key("gpuinfo");
    json_writer->StartObject();
    SetVal("name", devProps.name);
    SetVal("total_global_mem", int64_t(devProps.totalGlobalMem));
    SetVal("major", std::to_string(devProps.major));
    SetVal("minor", std::to_string(devProps.minor));
    SetVal("clock_rate", devProps.clockRate);
    SetVal("multi_processor_count", devProps.multiProcessorCount);

    int runtimeVersion, driverVersion;
    cudaRuntimeGetVersion(&runtimeVersion);
    cudaDriverGetVersion(&driverVersion);
    SetVal("driver_api", std::to_string(CUDA_VERSION));
    SetVal("driver_version", std::to_string(driverVersion));
    SetVal("runtime_version", std::to_string(runtimeVersion));
    SetVal("compute_version", std::to_string((devProps.major * 10 + devProps.minor)));
    json_writer->EndObject();
  }

  void Finalize(double postprocess_time, double total_time) {
    bool quiet = parameters->Get<bool>("quiet");
    int num_runs = parameters->Get<int>("num-runs");

    if (_process_times.size() > 1) {
      min_elapsed = *std::min_element(_process_times.begin(), _process_times.end());
      max_elapsed = *std::max_element(_process_times.begin(), _process_times.end());
      min_m_teps = (double)this->edges_visited / (max_elapsed * 1000.0);
      max_m_teps = (double)this->edges_visited / (min_elapsed * 1000.0);
    } else {
      min_elapsed = elapsed;
      max_elapsed = elapsed;
      min_m_teps = m_teps;
      max_m_teps = m_teps;
    }
    
    preprocess_time = parameters->Get<double>("preprocess-time");
    SetVal("process-times", process_times);
    SetVal("filtered-process-times", _process_times);
    SetVal("stddev-process-time", stddev_process_time);
    SetVal("avg-process-time", elapsed);
    SetVal("min-process-time", min_elapsed);
    SetVal("max-process-time", max_elapsed);
    SetVal("postprocess-time", postprocess_time);
    SetVal("total-time", total_time);
    SetVal("avg-mteps", m_teps);
    SetVal("min-mteps", min_m_teps);
    SetVal("max-mteps", max_m_teps);

    this->postprocess_time = postprocess_time;
    this->total_time = total_time;

    util::Sysinfo sysinfo;
    SetVal("sysinfo", sysinfo.getSysinfo());

    getGpuinfo();
    // util::Gpuinfo gpuinfo;
    // SetVal("gpuinfo", gpuinfo.getGpuinfo());

    util::Userinfo userinfo;
    SetVal("userinfo", userinfo.getUserinfo());

    // Add all the parameters to JSON
    this->parameters->List(*this);

    if (json_writer != NULL) json_writer->EndObject();
    
    if (!quiet) {
      DisplayStats();
    }
  }
};

}  // namespace util
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
