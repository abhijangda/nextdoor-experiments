# Directory Structure

Regardless of the level of contribution you would like to make, or develop towards, it is essential to understand the source directory structure of Gunrock. Knowing where things are will help decide the best place to invest your time in, and make the changes needed for your particular problem.

[gunrock's root]
  - **.github** --
  GitHub related files and folders, such as issue and pull request templates can be found here.

  - **cmake** --
  CMake modules are located here, allowing us to download googletest source code on the fly during cmake build, and other modules to look for the respective libraries and prerequisites.

  - **dataset** --
  Important gunrock datasets used in the publications, performance testing, correctness checking and unit tests are located here.
    - **large** --
    Large datasets are not stored in the github repository, we instead provide a Makefile to download the required dataset (or the entire collection) by simply running `make` in the respective subdirectory (or large directory for the whole collection).
      - [dataset]
    - **small** --
    Generally less than a 100 nodes (or a few 100s of nodes), these datasets are good candidates for correctness checking and quick tests. If you are developing a primitive within gunrock, it is recommended to use these for testing purposes and conduct a much more comprehensive tests/correctness check using the dataset under `large` directory.


  - **docker** --
  Docker files for different configurations, simply run `docker build . -t gunrock` in one of the subdirectories, and start the docker session using `nvidia-docker run -it gunrock /bin/bash`.

  - **docs** --
  Documentation repository (https://github.com/gunrock/docs) used for the main gunrock's website available at https://gunrock.github.io/docs/

  - **doxygen** --
  Doxygen generated API docs.

  - **examples** --
  Application test driver are located here in the subdirectory of the respective application. The `main()` function in each `test_<app>.cu` file in the subdirectories is responsible for setting the templated types and calling the CPU reference implementation as well as the Gunrock GPU implementation for the respective application.
    - [example application]

  - **externals** --
  External modules used by gunrock, perform `git submodule init && git submodule update` to fetch the modules; these include moderngpu, CUB and rapidJSON.

  - **gunrock** --
  Core of gunrock's source code is located here, this includes implementation of different algorithms/primitives, graph data structure, graph loader, operators, partitioner, other useful utlities, etc.

    - **app** --
    Implementations of each applications are located here (the driver is located under [root]/examples/[application name]).

      - **hello** --
      A sample application used to guide developers through writing their own application within gunrock.

        - hello_app.cu : High level view of the application, used to call the problem initialization and reset, enactor initialization and reset, and the validation routine. The external c-interface for the respective application is also located within the `<app_name>_app.cu` file.

        - hello_test.cuh : CPU reference code, validation routine and a display/output function are all located in the `<app_name>_test.cuh` file. Mainly used for correctness checking of the parallel implementation within gunrock and gunrock's programming model.

        - hello_problem.cuh : An application's data goes here. Declaration, initialization, reset and extraction of the data happens in the `<app_name>_problem.cuh` file.

        - hello_enactor.cuh : Iteration loop, operators and the actual algorithm implementation goes in the `<app_name>_enactor.cuh` file.


    - **graph** --
    Supported graph data structures and their helper functions can be found here. An example of such data structure is the Compressed Sparse Row format (CSR), and an example of a helper function for CSR format will be `GetNeighborListLength()`, which returns the number of neighbors for a given node.

    - **graphio** --
    Graph loader, labels loader, matrix market reader, randomly generated graphs (rgg) generator, etc.

    - **oprtr** --
    Gunrock's operators, such as advance (visiting neighbors), filter (removing nodes), for, neighborreduce, intersection are all located here in their respective folders. Load-balancing for these operators are also done within this directory as part of the operator kernels. See `oprtr_parameters.cuh` for the available parameters for the operators, and `oprtr.cuh` for the API for these operators.

    - **partitioner** --
    Graph partitioners such as metis, random, biased random partitioner are located here.

    - **util** --
    Extensive list of supporting utilities for gunrock. Device intrinsic primitives, array utilities, scan, reduction, and a lot more!


  - **python** --
  Simple Python bindings to the application's C Interfaces are located here as examples that can be built on. Feel free to create a python interface for an application that you would need.

  - **scripts** --
  Scripts for published papers, presentations and a for general performance testing are all located in the subdirectories.

  - **tools** --
  Useful scripts and tools to convert other graph file formats, such as snap or gr to matrix market format (mtx).

  - **unittests** --
  Googletest based unit testing framework for gunrock.
    - [unittest]
