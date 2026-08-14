// *******************************************************************
//      file with specific functions to solve Knapsack problem
// *******************************************************************
#ifndef _PROBLEM_H
#define _PROBLEM_H

//----------------- DEFINITION OF PROBLEM SPECIFIC TYPES -----------------------

struct TProblemData
{
    int n;                                      // size of the RKO vector 

    // other variables of the problem at hand
    int nItems;                                 // number of items
    int cap;                                    // capacity of the knapsack
    std::vector <int> w;                        // weigth of the items
    std::vector <int> b;                        // prize of the items
};


//-------------------------- FUNCTIONS OF SPECIFIC PROBLEM --------------------------


/************************************************************************************
 Method: ReadData
 Description: read the input data
*************************************************************************************/
void ReadData(char name[], TProblemData &data)
{ 
    FILE *arq;
    arq = fopen(name,"r");

    if (arq == NULL)
    {
        printf("\nERROR: File (%s) not found!\n",name);
        getchar();
        exit(1);
    }

    // => read data
    fscanf(arq, "%d", &data.nItems);
    fscanf(arq, "%d", &data.cap);
    
    //  weigth of items
    data.w.clear();
    data.w.resize(data.nItems);

    // prize of items
    data.b.clear();
    data.b.resize(data.nItems);

    for (int k=0; k<data.nItems; k++)
    {
        fscanf(arq, "%d", &data.b[k]);
        fscanf(arq, "%d", &data.w[k]);
    }
    
    // define the random-key vector size
    data.n = data.nItems;
}

/************************************************************************************
 Method: Decoder 
 Description: mapping the random-key solution into a problem solution
*************************************************************************************/
double Decoder(TSol &s, const TProblemData &data)
{   
    // create a solution of the KP
    std::vector <int> sol(data.n, 0);                        
    for (int i = 0; i < data.n; i++)
    {
        if (s.rk[i] > 0.5)
            sol[i] = 1;
    }

    // calculate the objective function value
    int cost = 0;
    int totalW = 0;
    for (int i = 0; i < data.n; i++)
    {
        if (sol[i] == 1)
        {
            cost += data.b[i];
            totalW += data.w[i];
        }
    }

    #define MAX(x,y) ((x)<(y) ? (y) : (x))

    // penalty infeasible solutions
    int infeasible = ((data.cap)<(totalW) ? (totalW - data.cap) : (0));
    cost = cost - (100000 * infeasible);

    // change to minimization problem
    cost = cost * -1;
    
    return cost;
}


/************************************************************************************
 Method: FreeMemoryProblem
 Description: Free local memory allocate by Problem
*************************************************************************************/
void FreeMemoryProblem(TProblemData &data){
    data.b.clear();
    data.w.clear();
}

#endif