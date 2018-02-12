#ifndef __K_MEANS_H
#define __K_MEANS_H

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <mpi.h>
#include <omp.h>
#include "kernel.h"

int k_means(double    **vectors,     			 //in:[numVectors][numDims] vectors from division of file
	double     *devVectors,				 //in:[numVectors*numDims]  pointer to vectors on GPU
	int        numDims,
	int        numVectors,
	int        numClusters,				 //k
	int        limit,   			 	 //max num of iterations dictated by file	
	int       *vectorToClusterRelevance, //out:[numVectors] for each vector states the cluster index to which it belongs
	double    **clusters,    			 //out:[numClusters][numDims] contains clusters centers
	MPI_Comm   comm);					 //communicator


cudaError_t copyVectorsToGPU(double **vectors,		//[numVectors][numDims] vectors from division of file
	double **devVectors,	//[numVectors*numDims]  pointer to vectors on GPU
	int numVectors,
	int numDims);

cudaError_t FreeVectorsOnGPU(double **devVectors);


#endif // !__K_MEANS_H