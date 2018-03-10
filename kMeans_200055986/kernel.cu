
#include "kernel.h"

__global__ void movePoints(double *devPoints, // points that were copied to device
	double *devSpeeds,			//speeds that were copied to device
	int numOfPoints,
	int numDims,
	int numThreadsInBlock,		//each thread takes care of one coord of one point
	double dt)					//the differencial for the change of coord
{
	int blockID = blockIdx.x;
	int i;

	//because we optimized num of blocks before calling the calling
	if ((blockID == gridDim.x - 1) && (numOfPoints % blockDim.x <= threadIdx.x)) { return; } //dismiss spare threads

	for (i = 0; i < numDims; i++)
	{
		devPoints[(blockID * numThreadsInBlock + threadIdx.x) * numDims + i] += devSpeeds[(blockID * numThreadsInBlock + threadIdx.x) * numDims + i] * dt;		
	}
}

__global__ void computeDistancesArray(double *devPoints,
	double *devClusters,
	int    numPoints,
	int    numClusters,
	int    numThreadsInBlock,
	int    numDims,
	double *devDistsPointsToClusters)
{
	int i;
	int blockID = blockIdx.x;
	double result = 0;

	if ((blockID == gridDim.x - 1) && (numPoints % blockDim.x <= threadIdx.x)) { return; } //dismiss spare threads

	//each thread computes a distance in a matrix of distances - the distance from each cluster center
	//no need to compute square root of result: (dist1 < dist2) <=> (sqrt(dist1) < sqrt(dist2))
	for (i = 0; i < numDims; i++)
	{
		result += (devPoints[(blockID * numThreadsInBlock + threadIdx.x) * numDims + i] - devClusters[threadIdx.y * numDims + i])
			* (devPoints[(blockID * numThreadsInBlock + threadIdx.x) * numDims + i] - devClusters[threadIdx.y * numDims + i]);
	}

	//this array contains for each point its distance from each cluster
	devDistsPointsToClusters[numPoints * threadIdx.y + (blockID * numThreadsInBlock + threadIdx.x)] = result;
}

__global__ void findMinDistanceForEachPointFromCluster(int numPoints,
	int    numClusters,
	int    numThreadsInBlock,
	double *devDistsPointsToClusters,
	int   *devPToCRelevance)
{
	int i;
	int blockId = blockIdx.x;
	double minIndex = 0;
	double minDistance, tempDistance;

	if ((blockIdx.x == gridDim.x - 1) && (numPoints % blockDim.x <= threadIdx.x)) { return; }  //dismiss spare threads

	minDistance = devDistsPointsToClusters[(numThreadsInBlock * blockId) + threadIdx.x];

	for (i = 1; i < numClusters; i++)
	{
		tempDistance = devDistsPointsToClusters[(numThreadsInBlock * blockId) + threadIdx.x + (i * numPoints)];
		if (minDistance > tempDistance)
		{
			minIndex = i;
			minDistance = tempDistance;
		}
	}

	devPToCRelevance[numThreadsInBlock * blockId + threadIdx.x] = minIndex;
}

cudaError_t movePointsWithCuda(double **points,	//cpu points that will be updated with new coords
	double *devPoints, 		//points that were copied to device
	double *devSpeeds,		//speeds that were copied to device
	int numOfPoints,
	int numDims,
	double dt)
{
	cudaError_t cudaStatus;
	cudaDeviceProp devProp; //used to retrieve specs from GPU

	int numBlocks, numThreadsInBlock;

	cudaGetDeviceProperties(&devProp, 0); // 0 is for device 0

	//if numOfPoints is bigger the maxThreadsPerBlock this is a problem
	//numThreadsInBlock = devProp.maxThreadsPerBlock / numOfPoints;
	numThreadsInBlock = devProp.maxThreadsPerBlock;
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

cudaError_t classifyPointsToClusters(double *devPoints,
	double **clusters,
	int     numPoints,
	int     numClusters,
	int		numDims,
	int    *pToCRelevance)
{
	cudaError_t cudaStatus;
	cudaDeviceProp devProp; //used to retrieve specs from GPU

	int numBlocks, numThreadsInBlock;

	cudaGetDeviceProperties(&devProp, 0); // 0 is for device 0

	//configuring kernel params
	numThreadsInBlock = devProp.maxThreadsPerBlock / numClusters;
	dim3 dim(numThreadsInBlock, numClusters);
	numBlocks = numPoints / numThreadsInBlock;

	if (numPoints % numThreadsInBlock > 0) { numBlocks++; }

	double *devClusters;
	double *devDistsPointsToClusters = 0;
	int   *devPToCRelevance = 0;

	cudaStatus = cudaMalloc((void**)&devClusters, numClusters * numDims * sizeof(double));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&devDistsPointsToClusters, numClusters * numPoints * sizeof(double));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&devPToCRelevance, numPoints * sizeof(int));
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
	computeDistancesArray <<<numBlocks, dim >>> (devPoints, devClusters, numPoints, numClusters, numThreadsInBlock, numDims, devDistsPointsToClusters);

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
	numBlocks = numPoints / numThreadsInBlock;
	if (numPoints % numThreadsInBlock > 0) { numBlocks++; }

	findMinDistanceForEachPointFromCluster <<<numBlocks, numThreadsInBlock >>> (numPoints, numClusters, numThreadsInBlock, devDistsPointsToClusters, devPToCRelevance);

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
	cudaStatus = cudaMemcpy(pToCRelevance, devPToCRelevance, numPoints * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


Error:
	cudaFree(devClusters);
	cudaFree(devDistsPointsToClusters);
	cudaFree(devPToCRelevance);

	return cudaStatus;
}
