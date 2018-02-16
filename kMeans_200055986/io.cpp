#include "io.h"

double* readVectorsFromFile(const char *fileName,           //file containing the data set
	int			numDims,
	int        *numVectors,         //number of given vectors in file
	int		   *numOfClustersToFind,
	int			*t,
	double		*dt,
	int        *iterationLimit,     //limit of k-means iteration allowed
	double     *qualityOfClusters,	//quality of clusters to find according to file
	double	   **pointsSpeeds)		//the speed for all points which we will read from the file

{
	int i, j, counter = 0;
	double *vectors;
	FILE *f;

	f = fopen(fileName, "r");
	assert(f != NULL);

	fscanf(f, "%d %d %d %lf %d %lf\n", numVectors, numOfClustersToFind, t, dt, iterationLimit, qualityOfClusters);
	
	//assiging the vectors Array (1D)
	vectors = (double*)malloc((*numVectors) * numDims * sizeof(double));
	assert(vectors != NULL);
	*pointsSpeeds = (double*)malloc((*numVectors) * numDims * sizeof(double));
	assert(*pointsSpeeds != NULL);
	for (i = 0; i < (*numVectors); ++i)
	{
		//read initial coordinates
		for (j = 0; j < numDims; ++j)
		{
			fscanf(f, "%lf ", &vectors[j + i* numDims]);
		}
		//read speeds
		for (j = 0; j < numDims; ++j)
		{
			fscanf(f, "%lf ", (*pointsSpeeds) + j + i* numDims);
		}

		fscanf(f, "\n");
	}
	
	fclose(f);
	return vectors;
}

void writeClustersToFile(char   *fileName,
	double **clusters,
	int     numClusters,
	int	 numDims,
	double   quality)
{
	int i, j;

	FILE *f = fopen(fileName, "w");
	assert(f != NULL);

	fprintf(f, "Number of clusters with the best measure:\n\n");
	fprintf(f, "K = %d QM = %.5f\n\n", numClusters, quality);
	fprintf(f, "Centers of the clusters:\n\n");
	
	for (i = 0; i < numClusters; ++i)
	{
		fprintf(f, "%c%d ", 'C', i + 1);

		for (j = 0; j < numDims; ++j)
		{
			fprintf(f, "%.2f ", clusters[i][j]);
		}

		fprintf(f, "\n\n");
	}
	fclose(f);
}

