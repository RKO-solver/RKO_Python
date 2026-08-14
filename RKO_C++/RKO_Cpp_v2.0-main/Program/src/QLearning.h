#ifndef _QLearning_H
#define _QLearning_H

/************************************************************************************
 Method: PrintPolicy
 Description: Print the policy in the screen
*************************************************************************************/
void PrintPolicy(std::vector<TState> S, int st)
{
    printf("\n\nPolicy\n");
    for (int i=0; i<(int)S.size(); i++)
    {
        printf("\n i = %d: \t", i);
        for (int j=0; j<(int)S[i].Ai.size(); j++)
        {
            printf("%d (%.2lf) \t", S[i].Ai[j], S[i].Qa[j]);
        }
    }
}

/************************************************************************************
 Method: CreatePossibleStates
 Description: Select all states s in S and specify actions a in A
*************************************************************************************/
void CreateStates(std::vector<std::vector<double>> parameters, int &numStates, int numPar, std::vector<TState> &S)
{
    #pragma omp critical
    { 
        // generate possible configurations
        numStates = parameters[0].size();
        for (int i=1; i<numPar; i++){
            numStates = numStates * parameters[i].size();
        }
        
        // create states
        for (int i=0; i<numStates; i++)
        {
            TState sAux;
            sAux.label = i; 
            sAux.par.resize(numPar);
            sAux.ci = 0;
            sAux.numN = 0;
            sAux.Ai.clear();
            sAux.Qa.clear();
            sAux.sumQ = 0;
            sAux.maxQ = 0;
            sAux.minA = 0;
            sAux.maxA = 0;
            
            S.push_back(sAux);
        }

        // define the parameter configuration of each state
        int nC = 1;
        for (int j=0; j<numPar; j++)
        {
            nC = nC * parameters[j].size();

            for (int i=0; i<numStates; i++)
            {
                int columnIndex = i / (numStates/nC);
                columnIndex = columnIndex % parameters[j].size();

                S[i].par[j] = parameters[j][columnIndex];
            }
        }

        // set of feasible control actions at each state
        for (int i=0; i<(int)S.size(); i++)
        {
            for (int j=0; j<(int)S.size(); j++)
            {
                // Calculate Hamming distance
                int distance = 0;
                for (int k=0; k<(int)S[i].par.size(); k++) {
                    if (S[i].par[k] != S[j].par[k]) {
                        distance++;
                    }
                }

                // We have an action from s_i if the new state is different from s_i by a maximmum of one parameter
                if (distance <= 1){
                    S[i].Ai.push_back(j);                       // define actions ai (index of the new state) from s_i
                    
                    double q0 = randomico(0.005,0.01);
                    S[i].Qa.push_back(q0);                      // initialize Q(si,ai)

                    if (q0 > S[i].maxQ)
                    {
                        S[i].maxQ = q0;
                        S[i].maxA = S[i].Ai.size()-1;
                    }
                }
            }
        }

        // print states
        // if (debug){    
        //     printf("\nNum States = %d",numStates);
        //     for (int i=0; i<numStates; i++)
        //     {
        //         printf("\n");
        //         printf("%d \t", S[i].label);
        //         for (int j=0; j<numPar; j++)
        //         {
        //             printf("%.2lf \t", S[i].par[j]);
        //         }
        //         // printf("\t %.2lf \t %d", S[i].ci, S[i].numN);
        //     }
        // }

        // print the transition matrix
        // if (debug)
        // {
        //     int st = 0;
        //     PrintPolicy(S, st);
        // }
    }
}

/************************************************************************************
 Method: ChooseAction()
 Description: Choose actions and update the parameters
*************************************************************************************/
int ChooseAction(const std::vector<TState> &S, int st, double epsilon)
{
    // ** choose action for current state from Q-Table using epsilon-Greedy policy
    int at;
                 
    // epsilon-greedy policy
    if (randomico(0,1) <= 1-epsilon) 
    {
        // choose the action with highest Q value
        at = S[st].maxA;
    }
    else
    {
        // choose a randonly selected action
        at = irandomico(0,S[st].Ai.size()-1);
    }

    // return the choose action
    return at;
}

/************************************************************************************
 Method: UpdateQLParameters(currentTime)
 Description: Update the parameters of the Q-Learning method
*************************************************************************************/
void SetQLParameter(float currentTime, int &Ti, int &restartEpsilon, float epsilon_max, float epsilon_min, double &epsilon, double &lf, double &df, int MAXTIME)
{
    // **** define epsilon ****
    static const double PI = 3.14159265;         

    // restart epsilon once Ti epochs are performed (Ti is 10% of the runtime)
    Ti = MAXTIME * 0.1;
    if (currentTime >= restartEpsilon * Ti){
        restartEpsilon++;

        // cosine decay with warm restart
        epsilon_max = epsilon_max - 0.1;
        if (epsilon_max < epsilon_min)
            epsilon_max = epsilon_min;
        epsilon = epsilon_max;
    }
    else {
        epsilon = epsilon_min + 0.5 * (epsilon_max - epsilon_min) * (1 + cos((((int)currentTime%Ti)/(float)(Ti))*PI));
    }
    
    // *** define learning rate ***

    // initialy, a higher priority is given to the newly gained information (exploration mode)
    // then, we decrement lf and have a higher priority for the existing information in Q-Table (exploitation mode)
    lf = 1 - (0.9 * currentTime / MAXTIME); 

    // *** define discount rate ***

    // we look for a higher, long-term reward
    df = 0.8;
}



#endif
