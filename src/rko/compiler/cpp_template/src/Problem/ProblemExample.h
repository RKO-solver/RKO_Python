// *******************************************************************
//      file with specific functions to solve a Problem
// *******************************************************************
#ifndef _PROBLEM_H
#define _PROBLEM_H

//----------------- DEFINITION OF PROBLEM SPECIFIC TYPES ----------------------------

struct TProblemData
{
    int n;                                      // size of the RKO vector 

    // other variables of the problem at hand
    
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
    

    fclose(arq);

    // define the size of the solution vector
    data.n = 1;
}


/************************************************************************************
 Method: Decoders
 Description: mapping the random-key solution into a problem solution
*************************************************************************************/
double Decoder(TSol &s, const TProblemData &data)
{
    // create a solution of the problem

    // calculate fitness
    s.ofv = 0;
    return s.ofv;
}

/************************************************************************************
 Method: FreeMemoryProblem
 Description: Free local memory allocate by Problem
*************************************************************************************/
void FreeMemoryProblem(TProblemData &data){}

#endif