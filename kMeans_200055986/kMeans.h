#ifndef __K_MEANS_H
#define __K_MEANS_H

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <mpi.h>
#include <omp.h>
#include "kernel.h"

void k_means(double    **points,     	//in:[numPoints][numDims] points from division of file
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

void movePointsWithOMP(double **points,		//[numPoints][numDims] points of each proc
	double **speeds,		//[numPoints][numDims] speeds for the points of each proc
	int numOfPoints,		//numOfPoints each proc
	int numDims,
	double dt);

void pickFirstKAsInitialClusterCenters(double** clusters, //[numClusters][numDims] contains clusters centers
	int k,
	double* points,			//[numPoints * numDims] all points we read from file
	int numOfPoints,		//total num of points from file
	int numDims);


#endif // !__K_MEANS_H