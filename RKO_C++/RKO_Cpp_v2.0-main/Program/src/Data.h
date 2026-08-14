#ifndef _DATA_H
#define _DATA_H

//------ DEFINITION OF TYPES OF RKO --------

/***********************************************************************************
 Struct: TSol
 Description: struct to represent a solution problem
************************************************************************************/
struct TSol
{
    std::vector <double> rk;                              // random-key vector
    double ofv = std::numeric_limits<double>::infinity(); // objetive function value

    double best_time = 0.0;                               // computational time to find the solution
    char nameMH[256];                                     // name of the metaheuristic that found the solution
};

/***********************************************************************************
 Struct: TRun
 Description: struct to variables use during the search process
************************************************************************************/
struct TRunData
{
    int strategy;                           // define the local search strategy (1 = first improvement; 2 = best improvement)
    int control;                            // define the control parameters (0 - offline; 1 - online)
    int MAXTIME;                            // define the maximum running time (stop condiction)
    int MAXRUNS;                            // maximum number of runs of the method
    int debug;                              // define the run mode (0 - save results in files; 1 - print results in screen)
    float restart;                          // define the restart strategy (0 - without restart; 1 - with restart)
    int sizePool;                           // define the size of the elite pool solutions
};

/***********************************************************************************
 Struct: TQ
 Description: struct to represent a quality matrix of Q-Learning
************************************************************************************/
struct TQ
{
    int S;                                  // state (parameter)
    double pVar;                            // value of the state
    double q;                               // q value
    int k;                                  // number of calls
    int kImp;                               // number of improvements
};

/***********************************************************************************
 Struct: TState
 Description: struct to represent a state of the MDP
************************************************************************************/
struct TState
{
    int label;                              // id of the state
    std::vector<double> par;                // vector of parameters        
    double ci;                              // cost of state i
    int numN;                               // number of runs of state i
    std::vector<int> Ai;                    // vector of actions
    std::vector<double> Qa;                 // vector of value function Q(s,a)
    double sumQ;                            // sum of Q(s,-)
    double maxQ;                            // max of Q(s,-)
    int minA;                               // index of the min Q(s,-)
    int maxA;                               // index of the max Q(s,-)
};

#endif