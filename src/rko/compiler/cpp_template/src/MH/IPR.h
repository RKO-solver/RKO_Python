#ifndef _IPR_H
#define _IPR_H

/************************************************************************************
 Method: IPR()
 Description: Apply the Implicit Path Relinking method
*************************************************************************************/
void IPR(const TRunData &runData, const TProblemData &data)
{
    const char* method = "IPR";

    float currentTime = 0;                          // computational time of the search process
    double start_timeMH = get_time_in_seconds();    // start computational time
    double end_timeMH = get_time_in_seconds();      // end computational time

    TSol atual;                                     // IPR current solution
    TSol guia;                                      // IPR guide solution
    TSol bestPath;                                  // best solution obtained by IPR
    TSol bestIteration;                             // best solution in each iteration
    TSol sCurrent;                                  // current solution in each iteration
    TSol sViz;                                      // neighboring solution in each iteration

    bestPath.ofv = std::numeric_limits<double>::infinity();
    bestPath.best_time = 0.0;

    // run the search process until stop criterion
    while (currentTime < runData.MAXTIME*runData.restart)
    {
        // randonly choose two elite solutions
        int k1, k2;
        do {
            k1 = irandomico(0,pool.size()-1);
            k2 = irandomico(0,pool.size()-1);
        }
        while (k1 == k2);

        TSol atual = pool[k1];                      
        TSol guia = pool[k2];                       

        TSol bestPath = atual;                 
        TSol bestIteration = atual;                
        TSol sCurrent = atual;                     
        TSol sViz = atual;                          

        int direction = 1;                          // internal (1) or external (-1) IPR

        int blockSize = data.n*0.10;                // block size
        int numBlock = data.n/blockSize;            // number of blocks
        std:: vector<int> fixedBlock(numBlock,1);   // binary vector indicating whether a block will be swapped (0) or not (1)
        int dist=0;                                 // number of different rk

        // calculates the difference between the solutions (measured by the number of different blocks)
        for (int i=0; i<data.n; i++)
        {
            if (atual.rk[i] != guia.rk[i]){
                fixedBlock[i%numBlock] = 0;
                dist++;
            }
        }

        // printf("\nGuia: %lf \t Atual: %lf [%d] >> ", guia.ofv, atual.ofv, dist);

        // if there is a difference between the solutions 
        if (dist > 0)
        {
            // 'zero out' the best solution on the path and carry out the search
            bestPath.ofv = INFINITY;
            int bestBlock=-1;

            // continue as long as there are blocks to exchange
            int numIteration = 0;
            while(numIteration < numBlock-1)
            {
                numIteration++;
                bestBlock=-1;
                bestIteration.ofv = INFINITY;

                // examine all possible blocks in a IPR iteration
                for(int i=0; i<numBlock; i++)
                {
                    if (fixedBlock[i] == 0){
                        int initialBlock = i*blockSize;
                        int finalBlock = initialBlock + blockSize;

                        // generate a neighbor of the current solution
                        sViz = sCurrent;
                        for (int k=initialBlock; k<finalBlock && k<data.n; k++)
                        {
                            // internal PR
                            if (direction == 1){
                                sViz.rk[k] = guia.rk[k];
                            }
                            
                            // external PR
                            else
                            if (direction == -1){
                                sViz.rk[k] = std::max(0.0, std::min(1.0 - guia.rk[k], 0.999999));
                            }

                        }
                        sViz.ofv = Decoder(sViz, data);
                        // printf("\n%.0lf, ",sViz.ofv);

                        // check if it is the best solution of the iteration
                        if(sViz.ofv < bestIteration.ofv){
                            bestIteration = sViz;
                            bestBlock = i;
                        }
                    }
                }

                // check if it is the best solution for the path
                if (bestIteration.ofv < bestPath.ofv)
                {
                    bestPath = bestIteration;
                    // printf("%.0lf, ",bestPath.ofv);
                }
                
                if (bestBlock >= 0){
                    // continue the search from the best solution of the iteration
                    sCurrent = bestIteration;

                    // fixed the index of the block changed in this iteration
                    fixedBlock[bestBlock] = 1;
                }
            }
        }

        // terminate the search process in MAXTIME
        end_timeMH = get_time_in_seconds();
        currentTime = end_timeMH - start_timeMH;

        // update the pool of solutions
        UpdatePoolSolutions(bestPath, method, runData.debug);
    }
}

#endif