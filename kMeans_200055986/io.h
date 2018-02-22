#ifndef __IO_H
#define __IO_H

#pragma warning( disable : 4996 )
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

double* readPointsDataFromFile(const char *fileName,  //file containing the data set
	int			numDims,
	int        *numPoints,			//number of given points in file
	int		   *numOfClustersToFind,
	int		   *t,
	double	   *dt,
	int        *iterationLimit,     //limit of k-means iteration allowed
	double     *qualityOfClusters,	//quality of clusters to find according to file
	double	   **pointsSpeeds);		//the speed for all points which we will read from the file
	
	

void writeClustersToFile(char    *fileName,
	double 		**clusters,		//[numClusters][numDims] cluster centers
	int  		numClusters,
	int		  	numDims,
	double   	quality);		//quality of the cluster group found

#endif //__IO_H