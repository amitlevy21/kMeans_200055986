#ifndef __KERNEL_H_
#define __KERNEL_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void computeDistancesArray(double *devVectors,					//[numVectors*numDims] vectors on GPU
	double *devClusters,					//[numClusters*numDims] clusters on GPU
	int    numVectors,
	int    numClusters,
	int    numThreadsInBlock,
	int    numDims,
	double *devDistsVectorsToClusters);	//[numClusters*numVectors] contains distances between vectors and clusters

__global__ void findMinDistanceForEachVectorFromCluster(int    numVectors,
	int    numClusters,
	int    numThreadsInBlock,
	double *devDistsVectorsToClusters,
	int   *devVToCRelevance);			//[numVectors] contains indexes of closest vector to cluster

cudaError_t computeClustersMeansWithCUDA(double *devVectors,		//in: [numVectors*numDims]
	double **clusters,			//in: [numClusters*numDims]
	int     numVectors,
	int     numClusters,
	int	 numDims,
	int    *vToCRelevance);	//out: [numVectors] contains indexes of closest vector to cluster

#endif // !__KERNEL_H_
