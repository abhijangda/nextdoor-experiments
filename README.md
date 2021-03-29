# Experimental Setup for NextDoor

## Introduction 

This repository contains GNN implementations for reproducing results in the paper "Accelerating Graph Sampling using GPUs" to appear at EuroSys'21. 
There are following GNN implementations in this repository:
* GraphSAGE in directory `GraphSAGE`
* GraphSAINT in directory `GraphSAINT`
* FastGCN and LADIES in directory `LADIES`
* ClusterGCN in directory `ClusterGCN`
* MVS GCN in directory `mvs_gcn`

## Prerequisites
<b>Linux Installation</b>: We recommend using Ubuntu 18.04 as the Linux OS. We have not tested our artifact with any other OS but we believe Ubuntu 20.04 should work too.

<b>Install Dependencies</b>: Execute following commands to install dependencies.

```
sudo apt update && sudo apt install gcc linux-headers-$(uname -r) make g++ git python-dev python3-dev wget unzip python-pip python3-pip cmake openmpi* libopenmpi* libmetis-dev
sudo pip3 install virtualenv
```

<b>Install CUDA</b>: We need to install CUDA before proceeding further. In our experiments we used CUDA 11.0 on Ubuntu 18.04. NextDoor uses other CUDA libraries like cuRAND and CUB. These libraries are not available as part of CUDA before version 11. Hence, we recommend using CUDA 11.0 because this will make the build process easier. CUDA 11.0 toolkit can be downloaded from https://developer.nvidia.com/cuda-11.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=runfilelocal
While installing CUDA please make sure that you install CUDA samples in your home directory and CUDA is installed in /usr/local/cuda. Alternatively, CUDA samples can be copied from /usr/local/cuda/samples/.

<b>Check CUDA Installation</b>: To check CUDA installation, go to CUDA samples installed in your home directory and execute following commands:
```
cd ~/NVIDIA_CUDA-11.0_Samples/1_Utilities/deviceQuery
make
./deviceQuery
```

Executing this deviceQuery command will show the information about GPUs in your system. If there is any error then either CUDA device is not present or CUDA driver is not installed correctly.

Set NVCC Path and CUDA libraries path: We assume that nvcc is present in /usr/local/cuda/bin/nvcc. Please make sure that this is a valid path and this nvcc is from CUDA 11.0 by using nvcc --version. Then export this in your PATH variable.
```
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
```
<b>Create Parent Directory for Evaluation</b>: 
For functionality and reproducibility create a top level directory that contains NextDoor and its experiments. Let say this directory is NextDoorEval.
```
mkdir NextDoorEval
export TOP_DIR=`pwd`/NextDoorEval  
export NEXTDOOR_DIR=$TOP_DIR/NextDoor
```
<b>Clone NextDoor and Download Dataset</b>: We must clone the repository and download  the Graph dataset from https://drive.google.com/file/d/19duY39ygWS3RAiqEetHUKdv-4mi_F7vj/view?usp=sharing . This zip file contains five graphs (PPI, Reddit, Orkut, Patents, LiveJournal1). Extract this zip file in input directory in the NextDoor’s repo clone directory. Following commands performs these operations:

```
cd $TOP_DIR
git clone --recurse-submodules https://github.com/plasma-umass/NextDoor.git
cd NextDoor
git submodule update --init --recursive
./downloadDataset.sh
unzip input.zip
```

## Reproducing Results

We will describe how to reproduce key results of paper in Figure 6, Figure 7, Figure 10, and Table 5. 
Setting Up Baselines
We will now set up baselines: KnightKing and existing GNNs.

### KnightKing
To setup KnightKing, clone from repository https://github.com/KnightKingWalk/KnightKing and compile using cmake. Below commands perform these operations:

```
cd $TOP_DIR
git clone https://github.com/KnightKingWalk/KnightKing.git --recurse-submodules
cd KnightKing
mkdir build && cd build
cmake .. && make -j
ls bin
```

Compiled binaries are in directory build/bin. This directory should contain three binaries: deepwalk, node2vec, and ppr.

### Existing GNNs 
We will now set up existing GNNs within the nextdoor-experiments repo. Execute below commands to clone nextdoor-experiments repo and setup GNNs.
```
cd $TOP_DIR
git clone https://github.com/abhijangda/nextdoor-experiments.git --recurse-submodules
cd nextdoor-experiments
./setup.sh
export NEXTDOOR_EXP_DIR=$TOP_DIR/nextdoor-experiments
```

To keep the packing simple we updated the implementation of these GNNs to use the latest version of NumPy, PyTorch and Tensorflow, so, that we have only a single CUDA version, i.e. CUDA 11.0. Unfortunately, with this new version they have added a restriction that each tensor must be of length less than 2GB. This is not possible for large graphs like Orkut and LiveJournal. Hence, for following GNNs we get Out of Memory error.
* GraphSAGE, MVS, and ClusterGCN gives out of Memory on Orkut and LiveJournal
* GraphSAINT, FastGCN, and LADIES gives out of Memory on Orkut

### Testing NextDoor

<b>Setting up Google Test</b>: We need to setup Google Test. Within the NextDoor directory, execute following commands to build googletest.
```
cd $NEXTDOOR_DIR/googletest
mkdir build          
cd build
cmake .. && make -j
```

<b>Building Tests</b>: To build all single GPU and multi GPU tests execute following command in NextDoor directory:

```
cd $NEXTDOOR_DIR
make -j
```

<b>Testing</b>: We recommend testing DeepWalk application before moving forward. To test an application, say DeepWalk, execute the following in the NextDoor directory.

```
cd $NEXTDOOR_DIR
./build/tests/singleGPU/deepwalk
```

This will show output like below:
```
[==========] Running 15 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 15 tests from DeepWalk
[ RUN      ] DeepWalk.LiveJournalTP
Graph Binary Loaded
Graph has 68555726 edges and 4847569 vertices
Using GPUs: [0,]
Final Size of each sample: 100
Maximum Neighbors Sampled at each step: 1
Number of Samples: 4847569
Maximum Threads Per Kernel: 4847616
Transit Parallel: End to end time 0.268283 secs
InversionTime: 0.0716658, LoadBalancingTime: 0, GridKernelTime: 0, ThreadBlockKernelTime: 0, SubWarpKernelTime: 0, IdentityKernelTime: 0
totalSampledVertices 172135742
checking results
Adj Matrix of Graph Created
[       OK ] DeepWalk.LiveJournalTP (17441 ms)
```
If any error comes please let us know.

### Performance Evaluation of Sampling Algorithms [Total Time: 4 Hours]
<b>System Requirements</b>:  For performance evaluation following requirements are needed.
* <i>CPU</i> and <i>RAM</i>: NextDoor and KnightKing do not require more than 10 GB of RAM. However, running existing GNNs on large graphs (patents, livejournal, and orkut) requires upto 40 GB of RAM. For performance evaluation against CPU baselines, (KnightKing and all GNNs), we used two 16-core Intel Xeon Silver CPUs.  Hence, we recommend that the reviewer do performance evaluation on similar systems. 
* <i>GPU</i>: In our paper we performed evaluation on NVIDIA Tesla V100 for single GPU results and 4 NVIDIA Tesla V100 for multiple GPU results. We also did performance evaluation on a NVIDIA GeForce GTX 1080Ti and found that performance dropped by a factor of 2 as compared to Tesla V100. In this case, We still obtained orders of magnitude improvement over CPU baselines. Hence, the evaluation of NextDoor vs CPU baselines (in Figure 7a and Figure 7b) will drop if a GPU with less execution resources than Tesla V100 is used. However, we still expect orders of magnitude improvement over CPU baselines and most of the other results in Figure 6 should show similar improvement over SP and TP. We require the GPU to have atleast 8 GB of available RAM.
* <i>Disk Space</i>: Our experiments will need atleast 30GB of diskspace.

We will now do performance evaluation of sampling implementations in NextDoor against the baselines to reproduce Figures 6 and 7 and Table 5. 
In some cases you can see more improvement than presented in the paper. This is because we have been working on a new and optimized implementation of NextDoor.

Before doing performance evaluation, (i) follow all steps of the “Prerequisite” section to clone NextDoor, and obtain datasets, (ii) follow all steps of “Setting Up Baselines” section to obtain the baselines, and (iii) follow all steps of “Testing NextDoor” section to build tests. We will use scripts available in nextdoor-experiments for benchmarking. Hence, all scripts must be executed in the order below.

<b>Reproduce Table 5</b>: We will reproduce end to end speedup experiments for GraphSAGE, FastGCN, and LADIES. Unfortunately, we couldn't get ClusterGCN integration to run before the submission deadline. For GraphSAGE we found that newer versions of Numpy and Tensorflow do not allow allocating tensors of size more than 2 GB. Hence, for Orkut and LiveJournal graph, we report OOM for GraphSAGE. We will update this in the final version of our paper.
This experiment will take upto 1 hour to execute.
Within the nextdoor-experiments directory execute below command and provide the absolute path to nextdoor.
```
cd $NEXTDOOR_EXP_DIR
python3 runEndToEnd.py -nextdoor $NEXTDOOR_DIR
```

<i>Results</i>: This script will print the results of evaluation.

<b>Reproduce Figure 6, and 7</b>: We will now reproduce the numbers in Figure 6, and 7. Following commands will execute KnightKing, existing GNNs, sampling applications in NextDoor and produce the numbers in a tabular format. There are two scripts to reproduce results.
First script is runGNNSampling.py . It executes all GNNs and stores the results. This command can take upto 2 hours and require upto 40 GB of RAM to do performance evaluation for large graphs (Orkut, Patents, and LiveJournal). Evaluation on small graphs (PPI and Reddit) take upto 10 minutes. To use only small graphs set -useSmallGraphs True command line argument.
```
cd $NEXTDOOR_EXP_DIR
python3 runGNNSampling.py -nextdoor $NEXTDOOR_DIR
```
Above script will produce `gnnSamplingResults.json` in nextdoor-experiments directory. <b>Make sure it exists before proceeding to next step.</b>

Second script is runBenchmarks.py, which  executes KnightKing, NextDoor on single GPU, and Nextdoor on multiple GPUs. By default NextDoor will be executed on GPU with ID 0. To use multiple GPUs, use `-gpu` argument to list all GPUs. Following is the example of this script. This script will take upto 30 mins:
```
cd $NEXTDOOR_DIR
make -j
cd $NEXTDOOR_EXP_DIR
python3 runBenchmarks.py -knightKing $TOP_DIR/KnightKing -nextdoor $NEXTDOOR_DIR -gpus 0,1,2,3
```
<b>Note that the above command assumes 4 GPUs with IDs 0,1,2,3. To use different numbers of GPUs please modify this list. If only one GPU is specified then multiple GPU results are not taken.</b>

<i>Reprint Previous Results</i>: Each invocation of this script stores the results in `benchmarkResults.json` file in `nextdoor-experiments` directory. To read these results and print them, use the optional flag `-printExistingResults true`.

<i>Results</i>: runBenchmarks.py will produce results for Figure 7a, 7b, 7c and Figure 6. If -useSmallGraphs True is used in runGNNSampling.py then results of Figure 7b will be produced only for PPI and Reddit graphs. With larger graphs the speedup of NextDoor over existing GNNs increases exponentially.

<b>Reproduce Figure 8 and Table 10</b>: We will now reproduce the numbers in Figure 8 and Table 10. These results are about four performance metrics. To obtain these results we will use script runPerfAnalysis.py in $NEXTDOOR_EXP_DIR. 
Script requires path to nvprof, which is usually available in `/usr/local/cuda/bin/nvprof`. Script will need <b>sudo</b> access because nvprof do not profile kernels  without <b>sudo</b> access. 

To obtain Figure 8(a) results for L2 Cache Transactions execute below command:
```
cd $NEXTDOOR_EXP_DIR
python3 runPerfAnalysis.py -nextdoor . $NEXTDOOR_DIR -nvprof /usr/local/cuda/bin/nvprof -metric l2_read_transactions
```

To obtain Figure 8(b) results for Warp Execution Efficiency, execute below commands:
```
cd $NEXTDOOR_EXP_DIR
python3 runPerfAnalysis.py -nextdoor  $NEXTDOOR_DIR -nvprof /usr/local/cuda/bin/nvprof -metric warp_execution_efficiency
```

To obtain Table 4 results for Global Store Efficiency, execute below commands:
```
cd $NEXTDOOR_EXP_DIR
python3 runPerfAnalysis.py -nextdoor  $NEXTDOOR_DIR -nvprof /usr/local/cuda/bin/nvprof -metric gst_efficiency
```

To obtain Table 4 results for Multiprocessor Activity, execute below commands:
```
cd $NEXTDOOR_EXP_DIR
python3 runPerfAnalysis.py -nextdoor  $NEXTDOOR_DIR -nvprof /usr/local/cuda/bin/nvprof -metric sm_efficiency
```
