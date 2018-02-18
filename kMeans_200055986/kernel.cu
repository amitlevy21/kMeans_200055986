
#include "kernel.h"

__global__ void movePoints(double *devPoints, // points that were copied to device
	double *devSpeeds,		//speeds that were copied to device
	int numOfPoints,
	int numDims,
	int numThreadsInBlock,	//each thread takes care of one coord of one point
	double dt)					//the differencial for the change of coord
{
	int blockID = blockIdx.x;
	int numOfCoord = numOfPoints * numDims;
	int i;
	double temp;
	double newTemp;

	if ((blockID == gridDim.x - 1) && (numOfPoints % blockDim.x <= threadIdx.x)) { return; } //dismiss spare threads

	for (i = 0; i < numDims; ++i)
	{
		temp = devPoints[(blockID * numThreadsInBlock + threadIdx.x) * numDims + i];
		devPoints[(blockID * numThreadsInBlock + threadIdx.x) * numDims + i] += devSpeeds[(blockID * numThreadsInBlock + threadIdx.x) * numDims + i] * dt;
		newTemp = devPoints[(blockID * numThreadsInBlock + threadIdx.x) * numDims + i];
		
	}
}

__global__ void computeDistancesArray(double *devVectors,
	double *devClusters,
	int    numVectors,
	int    numClusters,
	int    numThreadsInBlock,
	int    numDims,
	double *devDistsVectorsToClusters)
{
	int i;
	int blockID = blockIdx.x;
	double result = 0;

	if ((blockID == gridDim.x - 1) && (numVectors % blockDim.x <= threadIdx.x)) { return; } //dismiss spare threads

																							//each thread computes a distance in a matrix of distances
	for (i = 0; i < numDims; ++i)
	{
		result += (devVectors[(blockID*numThreadsInBlock + threadIdx.x)*numDims + i] - devClusters[threadIdx.y*numDims + i]) *  (devVectors[(blockID*numThreadsInBlock + threadIdx.x)*numDims + i] - devClusters[threadIdx.y*numDims + i]);
	}
	devDistsVectorsToClusters[numVectors*threadIdx.y + (blockID*numThreadsInBlock + threadIdx.x)] = result;
}

__global__ void findMinDistanceForEachVectorFromCluster(int    numVectors,
	int    numClusters,
	int    numThreadsInBlock,
	double *devDistsVectorsToClusters,
	int   *devVToCRelevance)
{
	int i;
	int xid = threadIdx.x;
	int blockId = blockIdx.x;
	double minIndex = 0;
	double minDistance, tempDistance;

	if ((blockIdx.x == gridDim.x - 1) && (numVectors % blockDim.x <= xid)) { return; }  //dismiss spare threads

	minDistance = devDistsVectorsToClusters[(numThreadsInBlock * blockId) + xid];

	for (i = 1; i < numClusters; ++i)
	{
		tempDistance = devDistsVectorsToClusters[(numThreadsInBlock * blockId) + xid + (i*numVectors)];
		if (minDistance > tempDistance)
		{
			minIndex = i;
			minDistance = tempDistance;
		}
	}

	devVToCRelevance[numThreadsInBlock*blockId + xid] = minIndex;
}

cudaError_t movePointsWithCuda(double **points,	//cpu points that will be updated with new coords
	double *devPoints, // points that were copied to device
	double *devSpeeds,		//speeds that were copied to device
	int numOfPoints,
	int numDims,
	double dt)
{
	cudaError_t cudaStatus;
	cudaDeviceProp devProp; //used to retrieve specs from GPU

	int numBlocks, numThreadsInBlock;

	cudaGetDeviceProperties(&devProp, 0); // 0 is for device 0

	numThreadsInBlock = devProp.maxThreadsPerBlock / numOfPoints;
	numBlocks = numOfPoints / numThreadsInBlock;
	
	if (numOfPoints % numThreadsInBlock > 0) { numBlocks++; }

	movePoints<<<numBlocks, numThreadsInBlock>>>(devPoints, devSpeeds, numOfPoints, numDims, numThreadsInBlock, dt);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	/* cudaDeviceSynchronize waits for the kernel to finish, and returns
	any errors encountered during the launch*/
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
	
	//update the points from gpu to cpu
	cudaStatus = cudaMemcpy((void**)points[0], devPoints, numOfPoints * numDims * sizeof(double), cudaMemcpyDeviceToHost);
	

Error:
	return cudaStatus;
}

cudaError_t computeClustersMeansWithCUDA(double *devVectors,
	double **clusters,
	int     numVectors,
	int     numClusters,
	int		numDims,
	int    *vToCRelevance)
{
	cudaError_t cudaStatus;
	cudaDeviceProp devProp; //used to retrieve specs from GPU

	int maxThreadsPerBlock;
	int numBlocks, numThreadsInBlock;

	cudaGetDeviceProperties(&devProp, 0); // 0 is for device 0

										  //configuring kerenl params
	numThreadsInBlock = devProp.maxThreadsPerBlock / numClusters;
	dim3 dim(numThreadsInBlock, numClusters);
	numBlocks = numVectors / numThreadsInBlock;

	if (numVectors % numThreadsInBlock > 0) { numBlocks++; }

	double *devClusters;
	double *devDistsVectorsToClusters = 0;
	int   *devVToCRelevance = 0;

	// Allocate GPU buffers for three vectors (two input, one output) 
	cudaStatus = cudaMalloc((void**)&devClusters, numClusters * numDims * sizeof(double));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&devDistsVectorsToClusters, numClusters * numVectors * sizeof(double));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&devVToCRelevance, numVectors * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	// Copy input from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(devClusters, clusters[0], numClusters * numDims * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	//launch kernels//
	computeDistancesArray << <numBlocks, dim >> > (devVectors, devClusters, numVectors, numClusters, numThreadsInBlock, numDims, devDistsVectorsToClusters);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	/* cudaDeviceSynchronize waits for the kernel to finish, and returns
	any errors encountered during the launch*/
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	//reconfiguring params for next kernel
	numThreadsInBlock = devProp.maxThreadsPerBlock;
	numBlocks = numVectors / numThreadsInBlock;
	if (numVectors % numThreadsInBlock > 0) { numBlocks++; }

	findMinDistanceForEachVectorFromCluster << <numBlocks, numThreadsInBlock >> > (numVectors, numClusters, numThreadsInBlock, devDistsVectorsToClusters, devVToCRelevance);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	/* cudaDeviceSynchronize waits for the kernel to finish, and returns
	any errors encountered during the launch*/
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(vToCRelevance, devVToCRelevance, numVectors * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


Error:
	cudaFree(devClusters);
	cudaFree(devDistsVectorsToClusters);
	cudaFree(devVToCRelevance);

	return cudaStatus;
}
