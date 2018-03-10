#include "io.h"

double* readPointsDataFromFile(const char *fileName,  //file containing the data set
	int			numDims,
	int        *numPoints,			//number of given points in file
	int		   *numOfClustersToFind,
	int		   *t,
	double	   *dt,
	int        *iterationLimit,     //limit of k-means iteration allowed
	double     *qualityOfClusters,	//quality of clusters to find according to file
	double	   **pointsSpeeds)		//the speed for all points which we will read from the file
{
	int i, j, counter = 0;
	double *points;
	FILE *f;

	f = fopen(fileName, "r");
	assert(f != NULL);

	fscanf(f, "%d %d %d %lf %d %lf\n", numPoints, numOfClustersToFind, t, dt, iterationLimit, qualityOfClusters);
	
	//assiging the points Array (1D)
	points = (double*)malloc((*numPoints) * numDims * sizeof(double));
	assert(points != NULL);
	*pointsSpeeds = (double*)malloc((*numPoints) * numDims * sizeof(double));
	assert(*pointsSpeeds != NULL);
	for (i = 0; i < (*numPoints); i++)
	{
		//read initial coordinates
		for (j = 0; j < numDims; j++)
		{
			fscanf(f, "%lf ", &points[j + i* numDims]);
		}
		//read speeds
		for (j = 0; j < numDims; j++)
		{
			fscanf(f, "%lf ", (*pointsSpeeds) + j + i* numDims);
		}

		fscanf(f, "\n");
	}
	fclose(f);
	return points;
}

void writeClustersToFile(char    *fileName,
	double **clusters,		//[numClusters][numDims] cluster centers
	int   numClusters,
	int		  numDims,
	double   quality)		//quality of the cluster group found
{
	int i, j;

	FILE *f = fopen(fileName, "w");
	assert(f != NULL);

	fprintf(f, "Number of clusters with the best measure:\n\n");
	fprintf(f, "K = %d QM = %.5f\n\n", numClusters, quality);
	fprintf(f, "Centers of the clusters:\n\n");
	
	for (i = 0; i < numClusters; i++)
	{
		fprintf(f, "%c%d ", 'C', i + 1);

		for (j = 0; j < numDims; j++)
		{
			fprintf(f, "%.2f ", clusters[i][j]);
		}

		fprintf(f, "\n\n");
	}
	fclose(f);
}

void printPoints(double *points, int numOfPoints, int numDims)
{
	int i, j;

	for (i = 0; i < numOfPoints; i++)
	{
		for (j = 0; j < numDims; j++)
		{
			printf("%lf ", points[i*numDims + j]);
			fflush(stdout);
		}

		printf("\n");
		fflush(stdout);
	}
}
