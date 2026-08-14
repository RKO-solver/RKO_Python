# RKO - Random-Key Optimizer v2.0
This is the second version of the RKO framework in C++.

This is an implementation of the RKO to solve combinatorial optimization problems. The code was prepared for Unix and Windows systems. However, the openMP paradigm needs to be enabled.

This algorithm's C++ code has been designed to be easy to reuse. Users can only implement specific functions (read and decoder in Problem.h). 

Here, we have the RKO version 2.0.1 code. RKO is constantly improving based on users’ feedback. 

![RKO_pipeline](https://github.com/user-attachments/assets/bb26650b-b5f0-4fc7-9cb1-55f5ca8b6132)


## References

When using this algorithm in academic studies, please refer to the following work:

[1] Chaves, A.A., Resende, M.G.C., Schuetz, M.J.A.,  Brubaker, J.K., Katzgraber, H.G., Arruda, E.F., Silva, R.M.A. 
A Random-Key Optimizer for Combinatorial Optimization. Journal of Heuristics, v. 31, n. 4, p. 32, 2025.

DOI
https://doi.org/10.1007/s10732-025-09568-z

Available here in technical report form: 
https://doi.org/10.48550/arXiv.2411.04293

## Scope

This code has been designed to solve the Knapsack Problem (KP). To solve other problems, users only need to configure the Problem.h file.


## Running the algorithm

* Enter the Program directory: `cd Program`
* Run the make command: `make rebuild`
* Run the RKO: `./runRKO ../Instances/KP/kp50.txt T`, where T is the maximum running time (in seconds)
* In Windows: runRKO.exe ../Instances/KP/kp50.txt T

* Or compile via terminal: `g++ -std=c++20 -o runRKO main.cpp -O3 -fopenmp`


## Code structure

The code structure is documented in [1] and organized in the following manner:

* **SPECIFIC_CODE:**
    * **Problem.h**: Contains data structure of the problem, the read data function, and the decoder.

* **GENERAL_CODE:**
    * **/MH**: Contains all of the metaheuristic (MH) algorithm's mechanisms.
    * **/Main**: Contains the main function to start the algorithm and stores the shared variables.
    * **Data.h**: Represents the data structures.
    * **Output.h**: Stores the output functions, including the best solution found and statistical analysis of the MH.

## File config_tests.conf is the configuration of the RKO test

* Metaheuristics in the RKO, each line is executed in a separate thread 
    * SA
    * ILS
    * VNS
    * BRKGA
    * BRKGA-CS
    * PSO
    * GA
    * LNS
    * GRASP
    * IPR

* defines the maximum number of runs
    * MAXRUNS 1

* defines the execution mode (0 for test mode or 1 for debug mode)
    * debug 1

* defines the parameter configuration mode (0 for offline tuning, 1 for online configuration using Q-Learning)
    * control 1

* local search strategy (1 for first improvement, 2 for best improvement)
    * strategy 1

* strategy to restart the search process (percentage of the maximum time at which the restart is triggered)
    * restart 1

* size of the elite pool solution
    * sizePool 10

Users need to create a folder named "Instances/ProblemName", where the instances must be; users also need to create a folder named "Results", where the results files are written.

## File Parameters

Users can choose two parameter settings: parameter tuning (option 0) and parameter control (option 1). 
 - For parameter tuning, users must inform the static configuration of each parameter in ParametersOffline.txt.
 - For parameter control, users must inform a set of possible values for each parameter in ParametersOnline.txt. We use the Q-Learning method to learn the best configuration for each metaheuristic during the search process.

## OpenMP

The code was implemented to run in parallel using the OpenMP directive. In this setup, #MH threads are required for each run. Each thread executes a different metaheuristic. The threads run independently, and information about the best solutions is shared through a solution pool.

## Object-oriented C++

An object-oriented C++ version of RKO has been recently developed by Johnny Dutra. The project was created from a fork of this repository and is available at the following link:

https://github.com/dutrajy/librko
