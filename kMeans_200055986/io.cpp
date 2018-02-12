#include "io.h"

double* readVectorsFromFile(const char *fileName,           //file containing the data set
	int			numDims,
	int		   *numOfClustersToFind,
	int        *numVectors,         //number of given vectors in file
	int        *iterationLimit,     //limit of k-means iteration allowed
	double      *qualityOfClusters) //quality of clusters to find according to file

{
	int i, j, counter = 0;
	double *vectors;
	FILE *f;

	f = fopen(fileName, "r");
	assert(f != NULL);

	fscanf(f, "%d %d %lf\n", numVectors, numOfClustersToFind, iterationLimit, qualityOfClusters);

	//assiging the vectors Array (1D)
	vectors = (double*)malloc((*numVectors) * numDims * sizeof(double));
	assert(vectors != NULL);
	for (i = 0; i < (*numVectors); ++i)
	{

		for (j = 0; j < numDims; ++j)
			fscanf(f, "%lf ", &vectors[j + i* numDims]);

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

