#ifndef __KERNEL_H_
#define __KERNEL_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void movePoints(double *devPoints, // points that were copied to device
	double *devSpeeds,		//speeds that were copied to device
	int numOfPoints,
	int numDims,
	int numThreadsInBlock,	//each thread takes care of one coord of one point
	double dt);				//the differencial for the change of coord

__global__ void computeDistancesArray(double *devPoints,	//[numPoints * numDims] Points on GPU
	double *devClusters,					//[numClusters * numDims] clusters on GPU
	int    numPoints,
	int    numClusters,
	int    numThreadsInBlock,
	int    numDims,
	double *devDistsPointsToClusters);		//[numClusters * numPoints] contains distances between Points and clusters

__global__ void findMinDistanceForEachPointFromCluster(int numPoints,
	int    numClusters,
	int    numThreadsInBlock,
	double *devDistsPointsToClusters,
	int   *devPToCRelevance);			//[numPoints] contains indexes of closest point to cluster

cudaError_t movePointsWithCuda(double **Points, //cpu points that will be updated with new coords
	double *devPoints, // points that were copied to device
	double *devSpeeds,		//speeds that were copied to device
	int numOfPoints,
	int numDims,
	double dt);

cudaError_t classifyPointsToClusters(double *devPoints,		//in: [numPoints * numDims]
	double **clusters,			//in: [numClusters * numDims]
	int     numPoints,
	int     numClusters,
	int	 	numDims,
	int    *pToCRelevance);	//out: [numPoints] contains indexes of closest point to cluster

#endif // !__KERNEL_H_
