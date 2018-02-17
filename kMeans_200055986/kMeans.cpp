#include "kMeans.h"

int k_means(double    **vectors,     			 //in:[numVectors][numDims] vectors from division of file
	double     *devVectors,				 //in:[numVectors*numDims]  pointer to vectors on GPU
	int        numDims,
	int        numVectors,
	int        numClusters,				 //k
	int        limit,   			 	 //max num of iterations dictated by file	
	int       *vectorToClusterRelevance, //out:[numVectors] for each vector states the cluster index to which it belongs
	double    **clusters,    			 //out:[numClusters][numDims] contains clusters centers
	MPI_Comm   comm)
{
	int      i, j, index, loop = 0;
	int		 sumDelta = 0;			// the sum of all the machines delta. indicates to stop the iteration
	int     *cudaVToCRelevance;		//[numvectors] used for CUDA kernels
	int     *newClusterSize;		//[numClusters]: no. vectors assigned in each new cluster                             
	int     *clusterSize;			//[numClusters]: temp buffer for MPI reduction 
	int      delta;					//num of vectors that changed their cluster relevance
	double  **newClusters;			//[numClusters][numDims] used to calculate new cluster means

									//initialize vectorToClusterRelevance[]
	for (i = 0; i < numVectors; ++i)
	{
		vectorToClusterRelevance[i] = -1;
	}

	//initializing memory 
	cudaVToCRelevance = (int*)malloc(numVectors * sizeof(int));
	assert(cudaVToCRelevance != NULL);

	newClusterSize = (int*)calloc(numClusters, sizeof(int));
	assert(newClusterSize != NULL);

	clusterSize = (int*)calloc(numClusters, sizeof(int));
	assert(clusterSize != NULL);

	newClusters = (double**)malloc(numClusters * sizeof(double*));
	assert(newClusters != NULL);
	newClusters[0] = (double*)calloc(numClusters * numDims, sizeof(double));
	assert(newClusters[0] != NULL);
	for (i = 1; i < numClusters; ++i) //arranging the rows
	{
		newClusters[i] = newClusters[i - 1] + numDims;
	}

	//start the k-means iterations
	do
	{
		delta = 0;

		computeClustersMeansWithCUDA(devVectors, clusters, numVectors, numClusters, numDims, cudaVToCRelevance);

		for (i = 0; i < numVectors; ++i)
		{
			//check if any vector changed his cluster
			if (vectorToClusterRelevance[i] != cudaVToCRelevance[i])
			{
				delta++;
				vectorToClusterRelevance[i] = cudaVToCRelevance[i];
			}

			//index = index of cluster that vector i now belongs to 
			index = cudaVToCRelevance[i];

			// update new cluster center: sum of vectors that belong to it 
			newClusterSize[index]++;
			for (j = 0; j < numDims; ++j)
				newClusters[index][j] += vectors[i][j];
		}

		//each proc shares his delta with others
		MPI_Allreduce(&delta, &sumDelta, 1, MPI_INT, MPI_SUM, comm);

		if (sumDelta == 0) { break; } //if no vector in the entire file changed his cluster - satisfies kMeans conditions

									  //sum all vectors in newClusters 
		MPI_Allreduce(newClusters[0], clusters[0], numClusters*numDims, MPI_DOUBLE, MPI_SUM, comm);
		MPI_Allreduce(newClusterSize, clusterSize, numClusters, MPI_INT, MPI_SUM, comm);

		//average the sum and replace old cluster centers with newClusters
		for (i = 0; i < numClusters; i++)
		{
			for (j = 0; j < numDims; j++)
			{
				if (clusterSize[i] > 1)
				{
					clusters[i][j] /= clusterSize[i];
				}
				newClusters[i][j] = 0.0;	//set back to 0 for next iterations
			}
			newClusterSize[i] = 0;			//set back to 0 for next iterations
		}

	} while (++loop < limit);

	//free all memory allocated
	free(newClusters[0]);
	free(newClusters);
	free(newClusterSize);
	free(clusterSize);
	free(cudaVToCRelevance);

	return 0;
}

cudaError_t copyVectorsToGPU(double **vectors, double **devVectors, double **vectorSpeeds, double **devSpeeds, int numVectors, int numDims)
{
	cudaError_t cudaStatus;

	//choosing GPU 0 as the device
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}

	//allocating memory on GPU for vectors
	cudaStatus = cudaMalloc((void**)devVectors, numVectors * numDims * sizeof(double));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		cudaFree(devVectors);
	}

	//allocating memory on GPU for vectors speeds
	cudaStatus = cudaMalloc((void**)devSpeeds, numVectors * numDims * sizeof(double));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		cudaFree(devVectors);
	}

	//copying the vectors from host to GPU
	cudaStatus = cudaMemcpy(*devVectors, vectors[0], numVectors * numDims * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		cudaFree(devVectors);
	}

	//copying the vectors speeds from host to GPU
	cudaStatus = cudaMemcpy(*devSpeeds, vectorSpeeds[0], numVectors * numDims * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		cudaFree(devVectors);
	}

	return cudaStatus;
}



cudaError_t FreeVectorsOnGPU(double **devVectors)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaFree(*devVectors);

	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaFree failed!");
	}
	return cudaStatus;
}