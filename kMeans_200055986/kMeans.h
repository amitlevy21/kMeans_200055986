#ifndef __K_MEANS_H
#define __K_MEANS_H

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <mpi.h>
#include <omp.h>
#include "kernel.h"

int k_means(double    **points,     	//in:[numPoints][numDims] points from division of file
	double     *devPoints,				//in:[numPoints * numDims]  pointer to points on GPU
	int        numDims,
	int        numPoints,
	int        numClusters,				//k
	int        limit,   			 	//max num of iterations dictated by file	
	int       *pointToClusterRelevance, //out:[numPoints] for each point states the cluster index to which it belongs
	double    **clusters,    			//out:[numClusters][numDims] contains clusters centers
	MPI_Comm   comm);					//communicator


cudaError_t copyPointDataToGPU(double **points,		//[numPoints][numDims] points that each proc has
	double **devPoints,			//[numPoints * numDims]  pointer to points on GPU
	double **pointSpeeds,		//[numPoints][numDims] point speeds of each proc
	double **devPointSpeeds,	//[numPoints * numDims]  pointer to speeds on GPU
	int numPoints,
	int numDims);

cudaError_t FreePointDataOnGPU(double **devPoints, double **devPointSpeeds);


#endif // !__K_MEANS_H