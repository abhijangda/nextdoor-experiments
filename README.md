# Experimental Setup for NextDoor

## Introduction 

This repository contains GNN implementations for reproducing results in the paper "Accelerating Graph Sampling using GPUs" to appear at EuroSys'21. 
There are following GNN implementations in this repository:
* GraphSAGE[1] in directory `GraphSAGE`
* GraphSAINT in directory `GraphSAINT`
* FastGCN and LADIES in directory `LADIES`
* ClusterGCN in directory `ClusterGCN`
* MVS GCN in directory `mvs_gcn`

## Benchmarking
Follow these steps to reproduce results in the paper.

### Prerequisites
<b> TODO Add this section from google doc </b>


### Clone NextDoor

Follow instructions in https://github.com/plasma-umass/nextdoor to obtain NextDoor, its datasets, and build all the sampling applications.

### Clone KnightKing

To setup KnightKing, clone from repository https://github.com/KnightKingWalk/KnightKing and compile using cmake. Below commands perform these operations:

```
git clone https://github.com/KnightKingWalk/KnightKing.git --recurse-submodules
cd KnightKing
mkdir build && cd build
cmake ..
make
ls build/bin
```

Compiled binaries are in directory build/bin. This directory should contain three binaries: deepwalk, node2vec, and ppr.

### Setup Existing GNNs

### Run Experiments

<b>Performance Evaluation of Sampling Application</b>

<b>End to End Performance Evaluation of Existing GNNs</b>

# References
<b>todo put references here</b>