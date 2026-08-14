#ifndef _SA_H
#define _SA_H

/************************************************************************************
 Method: SA
 Description: search process of the Simulated Annealing
*************************************************************************************/
void SA(const TRunData &runData, const TProblemData &data)
{
    const char* method = "SA";
    TSol s;                                  // current solution
    TSol sViz;                               // neighor solution
    TSol sBest;                              // best solution of SA

    double delta = 0;                        // difference between solutions
    double bestOFV;                          // value of the best solution in the current iteration

    double T0 = 0;                           // initial temperature
    double T = 0;                            // current temperature
    double alphaSA = 0;                      // cool rate
    int SAmax = 0;                           // number of iterations in a temperature T
    int IterT = 0;                           // iteracao corrente
    float betaMin = 0;                       // minimum perturbation
    float betaMax = 0;                       // maximum perturbation
    int reanneling = 0;                      // reanneling flag
    int improv = 0;                          // improvement flag
    float currentTime = 0;                   // computational time of the search process

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
    numPar = 5;
    std::vector<std::vector<double>> parameters;
    parameters.resize(numPar);

    readParameters(method, runData.control, parameters, numPar);

    // offline control
    if (runData.control == 0){
        // define the parameters of the SA
        SAmax   = parameters[0][0];
        alphaSA = parameters[1][0];
        betaMin = parameters[2][0];
        betaMax = parameters[3][0];
        T0      = parameters[4][0];
    }

    // online control
    else 
    {
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

            // define the initial parameters of the SA
            T0 = 1000000;
            SAmax = (int)S[iCurr].par[0];
            alphaSA = S[iCurr].par[1];
            betaMin = S[iCurr].par[2];
            betaMax = S[iCurr].par[3];
            T0      = S[iCurr].par[4];
        }
    }

    // Create the initial solution with random keys
    CreateInitialSolutions(s, data.n); 
    s.ofv = Decoder(s, data);
    sBest = s;

    // run the search process until stop criterion
    while (currentTime < runData.MAXTIME*runData.restart)
    {
        IterT = 0;
        if (!reanneling) T = T0;
        else T = T0*0.3;
        while (T > 0.0001 && currentTime < runData.MAXTIME*runData.restart)
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
                SAmax   = (int)S[iCurr].par[0];
                alphaSA = S[iCurr].par[1];     
                betaMin = S[iCurr].par[2];
                betaMax = S[iCurr].par[3];           
            }
            
            bestOFV = INFINITY;
            while (IterT < SAmax && currentTime < runData.MAXTIME*runData.restart)
            {
                if (stop_execution.load()) return;      
                
                IterT++;

                // Shake the current solution
                sViz = s;
                ShakeSolution(sViz, betaMin, betaMax, data.n);

                // calculate the OFV
                sViz.ofv = Decoder(sViz, data);
                
                // value function is the best solution found in this iteration
                if (sViz.ofv < bestOFV)
                    bestOFV = sViz.ofv;
                
                // calculate the delta SA
                delta = sViz.ofv - s.ofv;

                // define from which solution to continue the search
                if (delta < 0)
                {
                    // update current solution
                    s = sViz;

                    // update the best solution found by SA
                    if (s.ofv < sBest.ofv)
                    {
                        sBest = s;
                        improv = 1;

                        // update the pool of solutions
                        UpdatePoolSolutions(s, method, runData.debug);
                    }
                }
                else
                {
                    // metropolis criterion
                    double x = randomico(0,1);

                    if ( x < (exp(-delta/T)) )       
                        s = sViz;
                }
            } //End-SAmax

            // if (runData.debug)
            // printf("\nT: %lf \t current solution: %lf \t best solution: %lf", T, s.ofv, sBest.ofv);

            // Q-Learning 
            if (runData.control == 1){
                // The reward function is based on improvement of the current best fitness and binary reward
                if (improv){
                    R = 1;
                    improv = 0;
                }
                else{
                    R = (sBest.ofv - bestOFV)/bestOFV;
                }

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
            // *************************************************************

            T = T * alphaSA;
            IterT = 0;

            // apply local search
            sViz = s;
            // RVND(sViz, data, runData.strategy, RKorder);
            NelderMeadSearch(sViz, data);

            // update the best solution found by SA
            if (sViz.ofv < sBest.ofv)
            {
                sBest = sViz;

                // update the pool of solutions
                UpdatePoolSolutions(sBest, method, runData.debug);
            }

            // terminate the search process in MAXTIME
            end_timeMH = get_time_in_seconds();
            currentTime = end_timeMH - start_timeMH;
            
        } //Fim-T

        // reanneling
        reanneling = 1;
    }

    // print policy
    // if (runData.debug and runData.control == 1)
    //     PrintPolicy(S, st);
}

#endif