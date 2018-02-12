#include "kMeans.h"

cudaError_t copyVectorsToGPU(double **vectors, double **devVectors, int numVectors, int numDims)
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

	//copying the vectors from host to GPU
	cudaStatus = cudaMemcpy(*devVectors, vectors[0], numVectors * numDims * sizeof(double), cudaMemcpyHostToDevice);
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