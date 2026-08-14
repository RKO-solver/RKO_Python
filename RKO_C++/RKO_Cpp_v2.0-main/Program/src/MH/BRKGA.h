#ifndef _BRKGA_H
#define _BRKGA_H

/************************************************************************************
 Method: PARAMETRICUNIFORMCROSSOVER
 Description: create a new offspring with parametric uniform crossover
*************************************************************************************/
static TSol PUX(int eliteSize, int popSize, double rhoe, std::vector <TSol> &Pop, const int n)
{	
	TSol s;

    int eliteParent = irandomico(0, eliteSize - 1);                 // one chromosome from elite set
    int nonEliteParent = irandomico(eliteSize, popSize-1);          // one chromosome from nonelite population

	// create a new offspring
	s.rk.resize(n);

    // Mate
    for(int j = 0; j < n; j++)
    {
        //copy alelos of top chromossom of the new generation
        if (randomico(0,1) < rhoe){
            s.rk[j] = Pop[eliteParent].rk[j];
        }
        else{
            s.rk[j] = Pop[nonEliteParent].rk[j];
        }
    }

    return s;
}

/************************************************************************************
 Method: UpdatePopulationSize
 Description: update the size of the population according the current state
*************************************************************************************/
static void UpdatePopSize(int p, double pe, double pm, double rhoe, std::vector <TSol> &Pop, std::vector <TSol> &PopInter, const TProblemData &data)
{
    // *** define the new population size

    // size of the current population
    int oldPsize = Pop.size();

    // proportional pruning 
    if (oldPsize > p){

        // copy the current population
        PopInter = Pop;

        // define new size of Pop
        Pop.resize(p);

        // select the elite chromosomes
        for (int i=0; i<(int)(p*pe); i++){
            // copy p*pe best chromosomes
            Pop[i] = PopInter[i];
        }

        // select the non-elite chromosomes
        int pos = (int)(pe*oldPsize);
        for (int i=(int)(p*pe); i<p; i++){
            // copy the chromosome
            Pop[i] = PopInter[pos];
            pos++;
        }

        // clean intermediate population
        PopInter.clear();
        PopInter.resize(p);
    }
    
    // generate new chromosomes 
    else if (oldPsize < p){

        // define new size of Pop
        Pop.resize(p);

        // generate new chromosomes
        for (int k = oldPsize; k < p; k++)
        {
            if (stop_execution.load()) return;      
            
        	Pop[k] = PUX((int)(oldPsize*pe), oldPsize-1, rhoe, Pop, data.n);
            Pop[k].ofv = Decoder(Pop[k], data);
        }

        // sort new population
        sort(Pop.begin(), Pop.end(), sortByFitness);
        
        // clean intermediate population
        PopInter.clear();
        PopInter.resize(p);
    }
}

/************************************************************************************
 Method: BRKGA
 Description: search process of the BRKGA
*************************************************************************************/
void BRKGA(const TRunData &runData, const TProblemData &data)
{
    const char* method = "BRKGA";
    int p = 0;          	                    // size of population
    double pe = 0.0;              	            // fraction of population to be the elite-set
    double pm = 0.0;          	                // fraction of population to be replaced by mutants
    double rhoe = 0.0;              	        // probability that offspring inherit an allele from elite parent
 
    std::vector <TSol> Pop;                     // current population
    std::vector <TSol> PopInter;                // intermediary population
    TSol bestInd;                               // best individual found in past generation

    int numGenerations = 0;                     // number of generations
    int bestGeneration = 0;                     // generation in which found the best solution
    float currentTime = 0;                      // computational time of the search process
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
    numPar = 4;
    std::vector<std::vector<double>> parameters;
    parameters.resize(numPar);

    readParameters(method, runData.control, parameters, numPar);

    // offline control
    if (runData.control == 0){
        p    = parameters[0][0];
        pe   = parameters[1][0];
        pm   = parameters[2][0];
        rhoe = parameters[3][0];
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

            // define the initial parameters of the BRGKA
            p    = (int)S[iCurr].par[0];
            pe   = S[iCurr].par[1];                                                         
            pm   = S[iCurr].par[2];                                                    
            rhoe = S[iCurr].par[3];
        }
    }
    
    // initialize population
    Pop.clear(); 
    PopInter.clear(); 
    Pop.resize(p);
    PopInter.resize(p);

    // Create the initial chromosomes with random keys
    for (int i=0; i<p; i++)
    {
        CreateInitialSolutions(Pop[i], data.n); 
        Pop[i].ofv = Decoder(Pop[i], data);
        PopInter[i] = Pop[i];
    }
    
    // sort population in increase order of fitness
    sort(Pop.begin(), Pop.end(), sortByFitness);

    // save the best solution found
    bestInd = Pop[0];
    
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

            // execute action at
            iCurr = S[st].Ai[at];

            // define the parameters of the BRGKA according of the current state
            p       = (int)S[iCurr].par[0];
            pe      = S[iCurr].par[1];                                                         
            pm      = S[iCurr].par[2];                                                    
            rhoe    = S[iCurr].par[3]; 
            
            // update population size                                                 
            UpdatePopSize(p, pe, pm, rhoe, Pop, PopInter, data);                    
        }


        // The 'Pe' best chromosomes are maintained, so we just copy these into PopInter:
        for (int i=0; i<(int)(p*pe); i++){
            // copy the chromosome for next generation
            PopInter[i] = Pop[i]; 
        }  

        // We'll mate 'p - pe - pm' pairs; initially, i = pe, so we need to iterate until i < p - pm:
        TSol bestOff;
        bestOff.ofv = INFINITY;
        for (int i = (int)(p*pe); i < p - (int)(p*pm); i++){
            if (stop_execution.load()) return;      

            // Parametric uniform crossover
            PopInter[i] = PUX((int)(p*pe), p, rhoe, Pop, data.n);
 
            // Calculate the fitness of new chromosomes
            PopInter[i].ofv = Decoder(PopInter[i], data); 

            if (PopInter[i].ofv < bestOff.ofv){
                bestOff = PopInter[i];
            }
        }
        
        // We'll introduce 'pm' mutants:
        for (int i = p - (int)(p*pm) - (int)(p*pe); i < p; i++){
            if (stop_execution.load()) return;      
            
            CreateInitialSolutions(PopInter[i], data.n);
            PopInter[i].ofv = Decoder(PopInter[i], data); 
            // avoid calculate the fitness of the mutants
            // PopInter[i].ofv = INFINITY;             
        }  
                
        // Update the current population
        Pop = PopInter;   

        // Appy local search in one elite solution
        int pos = irandomico(0, (int)p*pe);
        NelderMeadSearch(Pop[pos], data);

        // Sort population in increase order of fitness
        sort(Pop.begin(), Pop.end(), sortByFitness);
        
        // We improve the best fitness in the current population 
        if (Pop[0].ofv < bestInd.ofv){
            bestGeneration = numGenerations;
            improv = 1;
            bestInd = Pop[0];
        
            // Update pool of solutions
            UpdatePoolSolutions(Pop[0], method, runData.debug);
        }

        // if (runData.debug) 
        // printf("\nGen: %d [%d %.2lf %.2lf %.2lf] \t best ind: %lf \t off: %lf", numGenerations, p, pe, pm, rhoe, bestInd.ofv, bestOff.ofv);

        // Q-Learning
        if (runData.control == 1){
            // We improve the best fitness in the current population 
            if (improv){
                // The reward function is based on improvement of the current best fitness and binary reward
                R = 1 + 1/p;                                        
                improv = 0;
            }    
            else{
                R = (bestInd.ofv - bestOff.ofv)/bestInd.ofv;
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

    // free memory of BRKGA components
    Pop.clear();
    PopInter.clear();

    // print policy
    // if (runData.debug and runData.control == 1)
    //     PrintPolicy(S, st);
}

#endif