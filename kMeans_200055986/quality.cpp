
#include "quality.h"
#include <stdio.h>

double euclidDistanceForQuality(int numDims,  //no. dimensions 
	double *p1,   	//[numDims] 
	double *p2)   	//[numDims] 
{
	int i;
	double dist = 0.0;

	for (i = 0; i < numDims; i++)
	{
		dist += (p1[i] - p2[i]) * (p1[i] - p2[i]);
	}

	return sqrt(dist);
}

double* computeClustersDiameters(double *points,
	int    numPoints,
	int    numClusters,
	int    numDims,
	int    *pToCR)
{
	double dist,				//Euclidian distance of two points from same cluster
		*diametersThreads,		//[numThreads * numClusters] collision free arrary - will contain the max distances from points that are in the same cluster calculated by each thread.
		*diameters;				//[numClusters] will contain the max distance between two points in the cluster
	int i, j, numThreads, tid, threadDiameterOffset;

	numThreads = omp_get_max_threads();

	diametersThreads = (double*)calloc(numThreads * numClusters, sizeof(double));

	diameters = (double*)malloc(numClusters * sizeof(double));

#pragma omp parallel for private(j, tid, dist, threadDiameterOffset) shared(diametersThreads)
	for (i = 0; i < numPoints; i++)
	{
		tid = omp_get_thread_num();
		threadDiameterOffset = tid * numClusters;

		for (j = i + 1; j < numPoints; j++)
		{

			if (pToCR[i] == pToCR[j]) //then these two points are in the same cluster
			{
				dist = euclidDistanceForQuality(numDims, points + (i * numDims), points + (j * numDims));
				if (dist > diametersThreads[threadDiameterOffset + pToCR[i]])
					diametersThreads[threadDiameterOffset + pToCR[i]] = dist;
			}
		}
	}

	//t0 computes max of diameters (sequential)
	for (i = 0; i < numClusters; i++)
	{
		diameters[i] = diametersThreads[i];
		for (j = 1; j < numThreads; j++)
		{
			if (diameters[i] < diametersThreads[j * numClusters + i])
				diameters[i] = diametersThreads[j * numClusters + i];
		}
	}

	free(diametersThreads);

	return diameters;
}

double computeClusterGroupQuality(double **clusters, int numClusters, int numDims, double *diameters)
{
	int i, j;

	//each cluster calculates the distances from all clusters but itself
	int numElements;  //the divider for the quality formula
	double quality = 0.0;
	double distance;

	if (numClusters == 1)
		numElements = 1;
	else
		numElements = numClusters * (numClusters - 1);
#pragma omp parallel for private(j) reduction(+ : quality)
	for (i = 0; i < numClusters; i++)
	{
		for (j = i + 1; j < numClusters; j++)
		{
			distance = euclidDistanceForQuality(numDims, clusters[i], clusters[j]);
			if(distance != 0)
				quality += (diameters[i] + diameters[j]);
		}
	}
	return quality / numElements;
}



