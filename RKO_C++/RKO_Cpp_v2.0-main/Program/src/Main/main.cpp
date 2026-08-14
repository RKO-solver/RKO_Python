// C++ library
#include <sys/time.h>
#include <math.h>
#include <cstring>
#include <ctime>
#include <iostream>
// Windows
#if defined(_WIN32) || defined(_WIN64)  
    #include <windows.h>
// Unix-like (Linux, macOS)
#else  
    #include <time.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string.h>
#include <algorithm>
#include <utility>  
#include <numeric>  
#include <map>
#include <limits>
#include <random>
#include <chrono>
#include <iomanip> 
#include <sstream> 
#include <fstream> 
#include <omp.h>
#include <atomic>

// RKO data
#include "../Data.h" 

// pseudo-random number generator Mersenne Twister
std::mt19937 rng;

// execution flag
std::atomic<bool> stop_execution(false);    

// pool of best solutions with diversity
std::vector <TSol> pool;                    

// RKO library
#include "../Problem/Problem.h"
#include "../Output.h"     
#include "../Method.h"
#include "../QLearning.h"
#include "../MH/BRKGA.h"
#include "../MH/SA.h"
#include "../MH/GRASP.h"
#include "../MH/ILS.h"
#include "../MH/VNS.h"
#include "../MH/PSO.h"
#include "../MH/GA.h"
#include "../MH/BRKGA_CS.h"
#include "../MH/LNS.h"
#include "../MH/IPR.h"
#include "../MH/MultiStart.h"                                    

// /************************************************************************************
// 								MAIN FUNCTION AREA
// *************************************************************************************/
int main(int argc, char *argv[ ])
{                          
    // run data
    TRunData runData;

    // name of the instance
    char nameInstance[256];  

    strncpy(nameInstance,argv[1],255);
    runData.MAXTIME = std::stoi(argv[2]); 

    // define the total number of metaheuristics available
    #define TOTAL_MH 11

    // name of the available metaheuristics
    const char *all_algorithms[TOTAL_MH] = {
        "BRKGA", "SA", "GRASP", "ILS", "VNS", "PSO", "GA", "LNS", "BRKGA-CS", "MultiStart", "IPR"
    };

    // Declare as funções com a assinatura correta
    void BRKGA(const TRunData &, const TProblemData &);
    void SA(const TRunData &, const TProblemData &);
    void GRASP(const TRunData &, const TProblemData &);
    void ILS(const TRunData &, const TProblemData &);
    void VNS(const TRunData &, const TProblemData &);
    void PSO(const TRunData &, const TProblemData &);
    void GA(const TRunData &, const TProblemData &);
    void LNS(const TRunData &, const TProblemData &);
    void BRKGA_CS(const TRunData &, const TProblemData &);
    void MultiStart(const TRunData &, const TProblemData &);
    void IPR(const TRunData &, const TProblemData &);

    // available metaheuristics
    void (*all_functions[TOTAL_MH])(const TRunData &, const TProblemData &) = {
        BRKGA, SA, GRASP, ILS, VNS, PSO, GA, LNS, BRKGA_CS, MultiStart, IPR
    };

    // dynamic arrays with chosen methods
    #define MAX_MH 100
    void (*functions_MH[MAX_MH])(const TRunData &, const TProblemData &);
    const char *algorithms[MAX_MH];
    int NUM_MH = 0;

    // Open the configuration file
    FILE *fileConf = fopen("config/config_tests.conf", "r");
    if (!fileConf) {
        printf("\nERROR: File config-tests.conf not found\n");
        exit(1);
    }

    char line[250];
    runData.MAXRUNS = 0; 
    runData.debug = 0; 
    runData.control = 0;

    while (fgets(line, sizeof(line), fileConf)) {
        // Remove newline character
        line[strcspn(line, "\n")] = '\0';

        // Check for metaheuristic names
        int idx = -1;
        for (int i = 0; i < TOTAL_MH; ++i) {
            if (strcmp(line, all_algorithms[i]) == 0) {
                idx = i;
            }
        }

        if (idx != -1) { // If it is an metaheuristic
            functions_MH[NUM_MH] = all_functions[idx];
            algorithms[NUM_MH] = all_algorithms[idx];
            NUM_MH++;
        }

        // Check for MAXRUNS
        if (strncmp(line, "MAXRUNS", 7) == 0) {
            sscanf(line, "MAXRUNS %d", &runData.MAXRUNS);
        }

        // Check for debug
        else if (strncmp(line, "debug", 5) == 0) {
            sscanf(line, "debug %d", &runData.debug);
        }

        // Check for control
        else if (strncmp(line, "control", 7) == 0) {
            sscanf(line, "control %d", &runData.control);
        }

        // Check for strategy
        else if (strncmp(line, "strategy", 8) == 0) {
            sscanf(line, "strategy %d", &runData.strategy);
        }

        // Check for restart
        else if (strncmp(line, "restart", 7) == 0) {
            sscanf(line, "restart %f", &runData.restart);
        }

        // Check for size pool
        else if (strncmp(line, "sizePool", 8) == 0) {
            sscanf(line, "sizePool %d", &runData.sizePool);
        }
    }
    fclose(fileConf);

    // input: read data of the instance problem
    TProblemData data;     
    ReadData(nameInstance, data);      
    
    double foBest = INFINITY,
           foAverage = 0.0;

    float timeBest = 0.0,
          timeTotal = 0.0;

    std::vector <double> ofvs;
    ofvs.clear();

    // best solutions found in MAXRUNS that is saved in out file
    TSol bestSolution;
    bestSolution.ofv = std::numeric_limits<double>::infinity();

    // run RKO MaxRuns for each instance                                        
    printf("\n\nInstance: %s \nRun: ", nameInstance);
    for (int run=0; run<runData.MAXRUNS; run++)
    {
        // current random seed
        int RSEED = 0;        

        // obatin a seed of the clock
        RSEED = std::chrono::steady_clock::now().time_since_epoch().count();

        // set new seed
        if (!runData.debug) rng.seed (RSEED);

        // use a fixed seed in debug mode
        if (runData.debug) rng.seed (1234);

        // runs
        printf("%d ", run+1);

        // best solution found in a run
        TSol bestSolutionRun;  
        bestSolution.best_time = 0.0;
        
        // computational times
        double start_time, end_time;                
        start_time = get_time_in_seconds();
        end_time = get_time_in_seconds();

        // ****************** run the RKO ******************
        // run all metaheuristics in parallel using OpenMP
        stop_execution.store(false);

        if (NUM_MH > 0){
            // create initial solutions in the pool of solutions
            pool.clear();
            pool.resize(runData.sizePool);
            CreatePoolSolutions(data, runData.sizePool);

            // best solution found in this run
            bestSolutionRun = pool[0];

            omp_set_num_threads(NUM_MH);
            #pragma omp parallel private(rng) shared(pool, stop_execution)
            {
                while (end_time - start_time < runData.MAXTIME)
                {
                    // Reset the cancellation flag
                    stop_execution.store(false);

                    #pragma omp for 
                    for (int i = 0; i < NUM_MH; ++i) { 
                        // checks the cancellation point
                        #pragma omp cancellation point for

                        // get the associated function
                        void (*function_mh)(const TRunData &, const TProblemData &) = functions_MH[i];

                        if (runData.debug) {
                            printf("\nThread %d executing MH_%d %s [%.2lf].", omp_get_thread_num(), i, algorithms[i], end_time - start_time);
                        }
                        
                        // calls the metaheuristic function
                        function_mh(runData, data);

                        // cancels when a thread ends
                        stop_execution.store(true);
                        #pragma omp cancel for
                    }
                    
                    // end running time
                    end_time = get_time_in_seconds();

                    // store the best solution found
                    if (pool[0].ofv < bestSolutionRun.ofv)
                        bestSolutionRun = pool[0];

                    // restart the pool of solutions in case of restart
                    if (end_time - start_time < runData.MAXTIME)
                        CreatePoolSolutions(data, runData.sizePool);
                }
            }
            // Reset the cancellation flag
            stop_execution.store(false);
        }

        // option not implemented
        else {
            printf("\n\nThis solver has not been implemented.\n");
            exit(1);
        }

        // store the best solution found in MAXRUNS
        if (bestSolutionRun.ofv < bestSolution.ofv)
            bestSolution = bestSolutionRun;

        // calculate best and average results
        if (bestSolutionRun.ofv < foBest)
            foBest = bestSolutionRun.ofv;

        foAverage += bestSolutionRun.ofv;

        // fitness of each solution found in the runs
        ofvs.push_back(bestSolutionRun.ofv);

        // computational time
        timeBest += bestSolutionRun.best_time - start_time;
        timeTotal += end_time - start_time;
    }

    // create a .csv file with average results
    foAverage = foAverage / runData.MAXRUNS;
    timeBest = timeBest / runData.MAXRUNS;
    timeTotal = timeTotal / runData.MAXRUNS;

    if (!runData.debug)
    {
        WriteSolution(algorithms, NUM_MH, bestSolution, timeBest, timeTotal, nameInstance, data);
        WriteResults(algorithms, NUM_MH, foBest, foAverage, ofvs, timeBest, timeTotal, nameInstance);
    }
    else
    {
        WriteSolutionScreen(algorithms, NUM_MH, bestSolution, timeBest, timeTotal, nameInstance, data, pool);
    }

    // Stable machine-readable integration output with full double precision.
    WriteMachineReadableResult(bestSolution);

    // free memory with problem data
    FreeMemoryProblem(data);

    return 0;
}
