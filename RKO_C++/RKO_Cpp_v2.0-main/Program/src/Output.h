/************************************************************************************
									IO Functions
*************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <math.h>
#include <limits>

#include "Data.h"

/************************************************************************************
 Machine-readable integration record.

 The legacy human output intentionally keeps its historical formatting.  This
 additive record uses max_digits10 so external integrations can reconstruct the
 exact random-key vector found by the optimizer.
*************************************************************************************/
void WriteMachineReadableResult(const TSol &s)
{
	const int precision = std::numeric_limits<double>::max_digits10;
	printf("\n@@RKO_RESULT_V1@@ %llu %.*g",
		   static_cast<unsigned long long>(s.rk.size()), precision, s.ofv);
	for (double key : s.rk)
		printf(" %.*g", precision, key);
	printf("\n");
	fflush(stdout);
}

/************************************************************************************
 Metodo: WriteSolutionScreen
 Description: Outputs the solution to the screen using the Decoder.
*************************************************************************************/
void WriteSolutionScreen(const char *algorithms[], int numMH, TSol s, 
						 float timeBest, float timeTotal, char instance[], 
						 const TProblemData &data, std::vector <TSol> pool)
{
	printf("\n\n\nRKO: ");
	for (int i=0; i<numMH; i++)
		printf("%s | ", algorithms[i]);
	printf("\nBest MH: %s \nInstance: %s \nsol: ", s.nameMH, instance);
	for (int i=0; i<data.n; i++)
		printf("%.3lf ", s.rk[i]);

	printf("\nofv: %.5lf", s.ofv); 
	printf("\nTotal time: %.3f",timeTotal);
	printf("\nBest time: %.3f\n\n",timeBest);

	// print solution pool 
	printf("\nSolution Pool:\n");
	for (int i = 0; i< (int)pool.size(); i++)
		printf("%.5lf [%s]\n", pool[i].ofv, pool[i].nameMH);
}

/************************************************************************************
 Metodo: WriteSolution
 Description: Outputs the solution in a txt file using the Decoder.
*************************************************************************************/
void WriteSolution(const char *algorithms[], int numMH, TSol s, 
				   float timeBest, float timeTotal, char instance[], 
				   const TProblemData &data)
{
	char name[256]="../Results/Solutions_RKO";
	strcat(name,".txt");

	// file to write the best solution found
	FILE *solFile;                              

    solFile = fopen(name,"a");

	if (!solFile)
	{
		printf("\n\nFile not found %s!!!",name);
		getchar();
		exit(1);
	}

    fprintf(solFile,"\n\nInstance: %s", instance);
	fprintf(solFile,"\nRKO: ");
	for (int i=0; i<numMH; i++)
		fprintf(solFile,"%s | ", algorithms[i]);

	fprintf(solFile,"\nSol: ");
	for (int i=0; i<data.n; i++)
		fprintf(solFile,"%.3lf ", s.rk[i]);

	fprintf(solFile,"\nofv: %lf", s.ofv);
  	fprintf(solFile,"\nBest time: %.3f",timeBest);
	fprintf(solFile,"\nTotal time:%.3f \n",timeTotal);

	fclose(solFile);
}

/************************************************************************************
 Metodo: WriteResults
 Description: Outputs the results in a csv file.
*************************************************************************************/
void WriteResults(const char *algorithms[], int numMH, double ofv, 
				  double ofvAverage, std::vector <double> ofvs, float timeBest, 
				  float timeTotal, char instance[])
{
	char name[256]="../Results/Results_RKO";
	strcat(name,".csv");

	FILE *File;
    File = fopen(name,"a");

	if (!File)
	{
		printf("\n\nFile not found %s!!!",name);
		exit(1);
	}

	fprintf(File,"\n%s\t", instance);
	for (int i=0; i<numMH; i++)
		fprintf(File,"%s | ", algorithms[i]);

    fprintf(File,"\t%d", (int)ofvs.size());
    for (unsigned int i=0; i<ofvs.size(); i++){
        fprintf(File,"\t%lf", ofvs[i]);   
	}
	fprintf(File,"\t%lf", ofv);
	fprintf(File,"\t%lf", ofvAverage);
	fprintf(File,"\t%.3f", timeBest);
	fprintf(File,"\t%.3f", timeTotal);

	fclose(File);
}
