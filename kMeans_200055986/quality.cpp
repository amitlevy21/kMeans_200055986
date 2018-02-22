
#include "quality.h"

double euclidDistanceForQuality(int numDims,  //no. dimensions 
	double *p1,   	//[numDims] 
	double *p2)   	//[numDims] 
{
	int i;
	double dist = 0.0;

	for (i = 0; i < numDims; ++i)
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
	double diameter, dist, *diametersThreads, *diameters;
	int i, j, numThreads, tid, stride;

	diameter = 0.0;

	numThreads = omp_get_max_threads();

	diametersThreads = (double*)calloc(numThreads  * numClusters, sizeof(double));

	diameters = (double*)malloc(numClusters * sizeof(double));

#pragma omp parallel for private(j, tid, dist, stride) shared(diametersThreads)
	for (i = 0; i < numPoints; ++i)
	{
		tid = omp_get_thread_num();
		stride = tid * numClusters;

		for (j = i + 1; j < numPoints; ++j)
		{
			if (pToCR[i] == pToCR[j])
			{
				dist = euclidDistanceForQuality(numDims, points + (i * numDims), points + (j * numDims));
				if (dist > diametersThreads[stride + pToCR[i]])
					diametersThreads[stride + pToCR[i]] = dist;
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
	return diameters;
}

double computeClusterGroupQuality(double **clusters, int numClusters, int numDims, double *diameters)
{
	int i, j;
	int numElements = 0; //the divider for the quality formula
	double quality = 0.0;

#pragma omp parallel for private(j) reduction(+ : quality)
	for (i = 0; i < numClusters; ++i)
	{
		for (j = i + 1; j < numClusters; ++j)
		{
			quality += (diameters[i] + diameters[j]) / euclidDistanceForQuality(numDims, clusters[i], clusters[j]);
			numElements++;
		}
	}
	return quality / numElements;
}



