#ifndef __IO_H
#define __IO_H

#pragma warning( disable : 4996 )
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

double* readVectorsFromFile(const char *fileName,           //file containing the data set
	int			numDims,
	int		   *numOfClustersToFind,
	int        *numVectors,         //number of given vectors in file
	int        *iterationLimit,     //limit of k-means iteration allowed
	double     *qualityOfClusters); //quality of clusters to find according to file
	

void writeClustersToFile(char    *fileName,
	double **clusters,		//[numClusters][numDims] cluster centers
	int   numClusters,
	int		  numDims,
	double   quality);		//quality of the cluster group found

#endif //__IO_H