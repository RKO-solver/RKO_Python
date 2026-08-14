#ifndef _VNS_H
#define _VNS_H

/************************************************************************************
 Method: VNS
 Description: search process of the Variable Neighborhood Search
*************************************************************************************/
void VNS(const TRunData &runData, const TProblemData &data)
{
    const char* method = "VNS";
    double beta = 0;                    // perturbation rate
    int Iter = 0;                       // current iteration
    int IterMelhora = 0;                // iteration that found the best solution
    int kMax = 0;                          // number of neighborhood strutures
    double betaMin = 0.0;               // intensity of perturbation
    int improv = 0;                     // improvement flag

    TSol s,                             // current solution
         sBest,                         // best solution of VNS
         sLine,                         // neighborhood solution
         sBestLine;                     // best neighborhood solution

    
    float currentTime = 0;              // computational time of the search process

    double start_timeMH = get_time_in_seconds();    // start computational time
    double end_timeMH = get_time_in_seconds();      // end computational time

    std::vector<int> RKorder;                   // define a order for the neighors
    RKorder.resize(data.n);
    std::iota(RKorder.begin(), RKorder.end(), 0);

    // Q-Learning parameters
    std::vector<TState> S;                      // finite state space
    int numPar = 0;                             // number of parameters
    int numStates = 0;                          // number of states
    int iCurr = 0;                              // current (initial) state
    double epsilon=0;                           // greed choice possibility
    double lf=0;                                // learning factor
    double df=0;                                // discount factor
    double R=0;                                 // reward
    std::vector <std::vector <TQ> > Q;          // Q-Table
    std::vector<int> ai;                        // actions
    float epsilon_max = 1.0;                    // maximum epsilon 
    float epsilon_min = 0.1;                    // minimum epsilon
    int Ti = 1;                                 // number of epochs performed
    int restartEpsilon = 1;                     // number of restart epsilon
    int st = 0;                                 // current state
    int at = 0;                                 // current action

    // ** read file with parameter values
    numPar = 2;
    std::vector<std::vector<double>> parameters;
    parameters.resize(numPar);

    readParameters(method, runData.control, parameters, numPar);

    // offline control
    if (runData.control == 0){
        kMax    = parameters[0][0];
        betaMin = parameters[1][0];
    }

    // online control
    else{
        // Q-Learning 
        if (runData.control == 1){
            // create possible states of the Markov chain
            CreateStates(parameters, numStates, numPar, S);

            // number of restart epsilon
            restartEpsilon = 1;  

            // maximum epsilon  
            epsilon_max = 1.0;  

            // current state
            iCurr = irandomico(0,numStates-1);

            // define parameters of VNS
            kMax    = S[iCurr].par[0];
            betaMin = S[iCurr].par[1];
        }
    }   

    // Create the initial solution with random keys
    sBest.ofv = INFINITY;
    for (int i=0; i<1; i++)
    {
        CreateInitialSolutions(s, data.n); 
        s.ofv = Decoder(s, data);
        if (s.ofv < sBest.ofv)
            sBest = s;
    }

    // current solution
    s = sBest;

    // run the search process until stop criterion           
    while (currentTime < runData.MAXTIME*runData.restart)
    {
        // Q-Learning 
        if (runData.control == 1){
            // set Q-Learning parameters  
            SetQLParameter(currentTime, Ti, restartEpsilon, epsilon_max, epsilon_min, epsilon, lf, df, runData.MAXTIME*runData.restart); 

            // choose a action at for current state st
            at = ChooseAction(S, st, epsilon);

            // execute action at of st
            iCurr = S[st].Ai[at];

            // define the parameters according of the current state
            kMax    = S[iCurr].par[0];
            betaMin = S[iCurr].par[1];              
        }

        // current neighborhood
        int k = 1;
        while (k <= kMax && currentTime < runData.MAXTIME*runData.restart)
        {
            if (stop_execution.load()) return;      
            
            Iter++;
            
            //s' <- perturb the best solution in the neighborhood k
            beta = randomico(k*betaMin,(k+1)*betaMin);

            // perturb the current solution (s)
            sLine = s;
            ShakeSolution(sLine, beta, beta, data.n);

            // calculate OFV
            sLine.ofv = Decoder(sLine, data);

            //s*' <- local search (s')
            sBestLine = sLine; 
            RVND(sBestLine, data, runData.strategy, RKorder);

            //s <- acceptance criterion (s,s*', historico)
            if (sBestLine.ofv < s.ofv)
            {
                s = sBestLine;
                
                // update the best solution found in VNS
                if (s.ofv < sBest.ofv){   
                    sBest = s;   

                    // return to the first neighborhood structure          
                    k = 1;
                    IterMelhora = Iter;
                    improv = 1;

                    // update the pool of solutions
                    UpdatePoolSolutions(sBestLine, method, runData.debug);
                }
            }
            else
            {
                // next neighborhood structure
                k++; 

                // metropolis criterion
                // if (randomico(0,1) < (exp(-(sBestLine.ofv - s.ofv)/(1000 - 1000*(currentTime / runData.MAXTIME*runData.restart)))) )
                // {
                //     s = sBestLine;
                // } 
            }

            // if (runData.debug) 
            // printf("\nIter: %d \t s'Best: %lf \t sBest: %lf \t sBestRun: %.2lf", Iter, sBestLine.ofv, sBest.ofv, pool[0].ofv);

            // terminate the search process in MAXTIME
            end_timeMH = get_time_in_seconds();
            currentTime = end_timeMH - start_timeMH;
        }

        // Q-Learning 
        if (runData.control == 1){
            // The reward function is based on improvement of the current best fitness and binary reward
            if (improv){
                R = 1;
                improv = 0;
            }
            else{
                R = (sBest.ofv - s.ofv)/s.ofv;
            }

            // if (runData.debug) printf("\t [%.4lf, %d, %d] \t [%d, %.2lf]", R, st, at, r, betaMin);

            // index of the next state
            int st_1 = S[st].Ai[at];

            // Update the Q-Table value
            S[st].Qa[at] = S[st].Qa[at] + lf*(R + df*S[st_1].maxQ - S[st].Qa[at]); 

            if (S[st].Qa[at] > S[st].maxQ)
            {
                S[st].maxQ = S[st].Qa[at];
                S[st].maxA = at;
            }

            // Define the new current state st
            st = st_1;
        }
    }

    // print policy
    // if (runData.debug and runData.control == 1)
    //     PrintPolicy(S, st);
}


#endif