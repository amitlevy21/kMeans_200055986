#include "kMeans.h"

int k_means(double    **points,     	//in:[numPoints][numDims] points from division of file
	double     *devPoints,				//in:[numPoints * numDims]  pointer to points on GPU
	int        numDims,
	int        numPoints,
	int        numClusters,				//k
	int        limit,   			 	//max num of iterations dictated by file	
	int       *pointToClusterRelevance, //out:[numPoints] for each point states the cluster index to which it belongs
	double    **clusters,    			//out:[numClusters][numDims] contains clusters centers
	MPI_Comm   comm)					//communicator
{
	int      i, j, index, loop = 0;
	int		 sumDelta = 0;			// the sum of all the procs delta. indicates to stop the iteration
	int     *cudaPToCRelevance;		//[numPoints] used for CUDA kernels
	int     *newClusterSize;		//[numClusters]: no. points assigned in each new cluster                             
	int     *clusterSize;			//[numClusters]: temp buffer for MPI reduction 
	int      delta;					//num of points that changed their cluster relevance
	double  **newClusters;			//[numClusters][numDims] used to calculate new cluster means

	//initialize pointToClusterRelevance[]
	for (i = 0; i < numPoints; ++i)
	{
		pointToClusterRelevance[i] = -1;
	}

	//initializing memory 
	cudaPToCRelevance = (int*)malloc(numPoints * sizeof(int));
	assert(cudaPToCRelevance != NULL);

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

		classifyPointsToClusters(devPoints, clusters, numPoints, numClusters, numDims, cudaPToCRelevance);

		for (i = 0; i < numPoints; ++i)
		{
			//check if any point changed his cluster
			if (pointToClusterRelevance[i] != cudaPToCRelevance[i])
			{
				delta++;
				pointToClusterRelevance[i] = cudaPToCRelevance[i];
			}

			//index = index of cluster that point i now belongs to 
			index = cudaPToCRelevance[i];

			newClusterSize[index]++;
			// update new cluster center: sum of points that belong to it 
			for (j = 0; j < numDims; ++j)
				newClusters[index][j] += points[i][j];
		}

		//each proc shares his delta with others
		MPI_Allreduce(&delta, &sumDelta, 1, MPI_INT, MPI_SUM, comm);

		if (sumDelta == 0) { break; } //if no point in the entire file changed his cluster - satisfies kMeans conditions

		//sum all points in newClusters 
		MPI_Allreduce(newClusters[0], clusters[0], numClusters * numDims, MPI_DOUBLE, MPI_SUM, comm);
		//ask moshe
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
	free(cudaPToCRelevance);

	return 0;
}

cudaError_t copyPointDataToGPU(double **points, double **devpoints, double **pointSpeeds, double **devSpeeds, int numpoints, int numDims)
{
	cudaError_t cudaStatus;

	//choosing GPU 0 as the device
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}

	//allocating memory on GPU for points
	cudaStatus = cudaMalloc((void**)devpoints, numpoints * numDims * sizeof(double));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		cudaFree(devpoints);
	}

	//allocating memory on GPU for points speeds
	cudaStatus = cudaMalloc((void**)devSpeeds, numpoints * numDims * sizeof(double));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		cudaFree(devpoints);
	}

	//copying the points from host to GPU
	cudaStatus = cudaMemcpy(*devpoints, points[0], numpoints * numDims * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		cudaFree(devpoints);
	}

	//copying the points speeds from host to GPU
	cudaStatus = cudaMemcpy(*devSpeeds, pointSpeeds[0], numpoints * numDims * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		cudaFree(devpoints);
	}

	return cudaStatus;
}



cudaError_t FreePointDataOnGPU(double **devPoints, double **devPointSpeeds)
{
	cudaError_t cudaStatus;

	cudaStatus = cudaFree(*devPoints);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaFree failed!");
	}

	cudaStatus = cudaFree(*devPointSpeeds);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaFree failed!");
	}

	return cudaStatus;
}

void movePointsWithOMP(double **points, double **speeds, int numOfPoints, int numDims, double dt)
{
	int i, j;

#pragma omp parallel for private(j)
	for (i = 0; i < numOfPoints; i++)
	{
		for (j = 0; j < numDims; j++)
		{
			points[i][j] += speeds[i][j] * dt;
		}
	}
}

void pickFirstKAsInitialClusterCenters(double** clusters, int k, double* points, int numOfPoints, int numDims)
{
	int i, j;

	for (i = 0; i < k; ++i)
	{
		for (j = 0; j < numDims; ++j)
		{
			clusters[i][j] = points[j + i * numDims];
		}
	}
}