#ifndef _GA_H
#define _GA_H

/************************************************************************************
 Method: UpdatePopulationSize
 Description: update the size of the population according the current state
*************************************************************************************/
static void UpdatePopulation(int sizePop, std::vector <TSol> &Pop, const TProblemData &data)
{
    // *** define the new population size

    // size of the current population
    int oldPsize = Pop.size();

    // pruning 
    if (oldPsize > sizePop){

        // define new size of Pop
        Pop.resize(sizePop);
    }
    
    // generate new chromosomes 
    else if (oldPsize < sizePop){
        // define new size of Pop
        Pop.resize(sizePop);

        for (int k = oldPsize; k < sizePop; k++)
        {     
            CreateInitialSolutions(Pop[k], data.n); 
            Pop[k].ofv = Decoder(Pop[k], data);  
        }
    }
}

/************************************************************************************
 Method: GA
 Description: search process of the Genetic Algorithm
*************************************************************************************/
void GA(const TRunData &runData, const TProblemData &data)
{
    const char* method = "GA";
    int sizePop = 0;                            // population size
    float probCros = 0.0;                       // crossover probability
    float probMut = 0.0;                        // mutation probability

    std::vector <TSol> Pop;                     // current population
    std::vector <TSol> PopInter;                // selected individuals to evolutionary process
    std::vector <TSol> PopNew;                  // offsprings

    TSol bestInd;                               // best solution found in SGA

    // useful local variables
    int numGenerations = 0;                     // number of generations
    float currentTime = 0;                      // computational time of the search process
    int bestGeneration = 0;                     // generation in which found the best solution
    int improv = 0;                             // improvement flag

    double start_timeMH = get_time_in_seconds();// start computational time
    double end_timeMH = get_time_in_seconds();  // end computational time

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
    numPar = 3;
    std::vector<std::vector<double>> parameters;
    parameters.resize(numPar);

    readParameters(method, runData.control, parameters, numPar);

     // offline control
    if (runData.control == 0){
        // define parameters of SGA
        sizePop  = parameters[0][0];
        probCros = parameters[1][0];
        probMut  = parameters[2][0];
    }

    // online control
    else {  
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

            // define parameters of SGA
            sizePop  = (int)S[iCurr].par[0];
            probCros = S[iCurr].par[1];
            probMut  = S[iCurr].par[2]; 
        }
    }

    // initialize population
    Pop.clear(); 
    PopNew.clear();
    PopInter.clear(); 

    Pop.resize(sizePop);
    PopNew.resize(sizePop);
    PopInter.resize(sizePop);

    // Create the initial chromosomes with random keys
    bestInd.ofv = 9999999999; 
    for (int i=0; i<sizePop; i++)
    {
        CreateInitialSolutions(Pop[i], data.n); 
        Pop[i].ofv = Decoder(Pop[i], data);  

        // set the best individual
        if (Pop[i].ofv < bestInd.ofv)
            bestInd = Pop[i];
    }
    
    // run the evolutionary process until stop criterion
    while (currentTime < runData.MAXTIME*runData.restart)
    {
    	// number of generations
        numGenerations++;
        
        // Q-Learning 
        if (runData.control == 1){
            // set Q-Learning parameters  
            SetQLParameter(currentTime, Ti, restartEpsilon, epsilon_max, epsilon_min, epsilon, lf, df, runData.MAXTIME*runData.restart); 

            // choose a action at for current state st
            at = ChooseAction(S, st, epsilon);

            // execute action at of st
            iCurr = S[st].Ai[at];

            // define the parameters according of the current state
            sizePop  = (int)S[iCurr].par[0];
            probCros = S[iCurr].par[1];
            probMut  = S[iCurr].par[2]; 

            UpdatePopulation(sizePop, Pop, data);
        }

        // Select parents using the TOURNAMENT selection method
        PopInter.clear();
        PopInter.resize(sizePop);
        for (int j=0; j<sizePop; j++)
        {
            int p1, p2, p3, melhor;

            do{
                p1 = irandomico(0,sizePop-1);
                p2 = irandomico(0,sizePop-1);
                p3 = irandomico(0,sizePop-1);
            }while (p1 == p2 || p1 == p3 || p2 == p3);
            
            if ((Pop[p1].ofv < Pop[p2].ofv) && (Pop[p1].ofv < Pop[p3].ofv))
                melhor = p1;
            else
            if (Pop[p2].ofv < Pop[p3].ofv)
                melhor = p2;
            else
                melhor = p3;

            // insert the best individual into the intermediate population 
            PopInter[j] = Pop[melhor];
        }

        // perform uniform crossover between the selected pairs of parents
        double avrgOff = 0.0;
        double bestOFcurrent = INFINITY;
        PopNew.clear();
        PopNew.resize(sizePop);
        for (int i=0; i<sizePop-1; i=i+2)
        {
            if (stop_execution.load()) return;      

            PopNew[i] = PopInter[i];
            PopNew[i+1] = PopInter[i+1];

            // check the probability of crossover
            if (randomico(0,1) < probCros)
            {
                for(int j = 0; j < data.n; j++)
                {
                    // swap the alleles of offspring i
                    if (randomico(0,1) < 0.5){
                        PopNew[i].rk[j] = PopInter[i+1].rk[j];
                    }

                    // swap the alleles of offspring i+1
                    if (randomico(0,1) < 0.5){
                        PopNew[i+1].rk[j] = PopInter[i].rk[j];
                    }

                    // check the probability of making a mutation in each gene of offspring i
                    if (randomico(0,1) <= probMut){
                        PopNew[i].rk[j] = randomico(0,1); 
                    }

                    // check the probability of making a mutation in each gene of offspring i+1
                    if (randomico(0,1) <= probMut){
                        PopNew[i+1].rk[j] = randomico(0,1); 
                    }
                }

                // calculate fitness
                PopNew[i].ofv   = Decoder(PopNew[i], data);
                PopNew[i+1].ofv = Decoder(PopNew[i+1], data);
            }

            avrgOff += PopNew[i].ofv + PopNew[i+1].ofv;

            // set the best individual
            if (PopNew[i].ofv < bestInd.ofv){
                bestInd = PopNew[i];
                bestGeneration = numGenerations;
                improv = 1;

                // update pool of solutions
                UpdatePoolSolutions(bestInd, method, runData.debug);
            }

            if (PopNew[i+1].ofv < bestInd.ofv){
                bestInd = PopNew[i+1];
                bestGeneration = numGenerations;
                improv = 1;

                // update pool of solutions
                UpdatePoolSolutions(bestInd, method, runData.debug);
            }

            // set the best offspring
            if (PopNew[i].ofv < bestOFcurrent){
                bestOFcurrent = PopNew[i].ofv;
            }
            if (PopNew[i+1].ofv < bestOFcurrent){
                bestOFcurrent = PopNew[i+1].ofv;
            }
        }

        // Apply local search in randomly selected solution and assigning it in Pop
        int pos1 = irandomico(0,sizePop-1);
        NelderMeadSearch(PopNew[pos1], data);

        // set the best individual
        if (PopNew[pos1].ofv < bestInd.ofv){
            bestInd = PopNew[pos1];
            bestGeneration = numGenerations;

            // update pool of solutions
            UpdatePoolSolutions(bestInd, method, runData.debug);
        }
           
        // replace the population with offspring
        Pop = PopNew;

        // if (runData.debug) 
        // printf("\nGen: %d \t sBest: %lf \t average: %lf", numGenerations, bestInd.ofv, avrgOff/sizePop);

        // Q-Learning 
        if (runData.control == 1){
            // The reward function is based on improvement of the current best fitness and binary reward
            if (improv){
                R = 1;
                improv = 0;
            }
            else{
                R = (bestInd.ofv - bestOFcurrent)/bestOFcurrent;
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

        // terminate the evolutionary process in MAXTIME
        end_timeMH = get_time_in_seconds();
        currentTime = end_timeMH - start_timeMH;
    }

    // free memory of GA components
    Pop.clear();
    PopNew.clear();
    PopInter.clear();

    // print policy
    // if (runData.debug and runData.control == 1)
    //     PrintPolicy(S, st);
}

#endif