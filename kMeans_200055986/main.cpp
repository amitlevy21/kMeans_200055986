#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "io.h"
#include "kMeans.h"
#include "kernel.h"
#include "quality.h"

#define DATA_FROM_FILE_SIZE 4


void createVectorAssignmentToMachinesArray(int totalNumPoints,
	int  numOfMachines,
	int  numDims,
	int *sendCounts,
	int *displs);

void movePointsWithOMP(double **points, double **speeds, int numOfPoints, int numDims, double dt);
void printPoints(double *points, int numOfPoints, int numDims);

int main(int argc, char *argv[])
{
	//variables required for MPI initialization
	int numprocs, myid;

	//variables for program
	int i, j, k,					// k is actual num of clusters
		continueKMeansFlag,			// 1 = continue, 0 = stop
		numDims = 2,
		totalNumVectors,			//num of vectors read from file
		iterationLimit,				//limit of iterations allowed in kMeans algorithm
		*vectorToClusterRelevance,  //[totalNumVectors] in use by p0 only! index is vector and value is cluster index which the vector belongs to
		*vToCRelevanceEachProc,		//[numVectorInMachine] in use for the K-Means only! holds the relevance of vectors to clusters for each machine
		*sendCounts = NULL,			//[numprocs] used for scatterV of vectors to machines
		*recvCounts = NULL,			//[numprocs] used for gatherV of vectors to cluster relevancies from all machines to p0
		*displsScatter = NULL,		//[numprocs] used for scatterV of vectors to machines
		*displsGather = NULL,		//[numprocs] used for gatherV of vectors to cluster relevancies from all machines to p0
		numVectorsInMachine,		//for each proc states how many vectors were assigned 
		*dataFromFile,
		t;							//maximum time for running the algorithm
		

	double	 *vectorsReadFile,		//[totalNumVectors * numDims] all vectors from data-set
		*devVectors = NULL,			//pointer to vectors on GPU 
		*devVectorSpeeds = NULL,	//pointer to vector speeds on GPU
		*diameters,				//[numClusters] in use by p0 only. contains diameters for each cluster
		**clusters,				//[k][numDims] contains cluster centers
		requiredQuality,		//required quality to reach stated in input file
		clusterGroupQuality,	//quality computed for each group of k clusters
		**vectorsEachProc,		//[numVectorsInMachine][numDims] holds all vectors that were assigned to a proc
		*pointsSpeeds,			//[totalNumVectors * numDims] holds all speeds for each points
		**pointsSpeedsEachProc, //[numVectorsInMachine][numDims] holds the speeds for the points the belong to each proc
		dt,						//the change in time in each iteration
		currentT = 0;				//the current time for each proc
		
	char    *inputFile = "C:\\data_kmeans.txt",
		*outputFile = "C:\\data_result.txt";

	//establishing MPI communication
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Status status;

	continueKMeansFlag = 1; //continue iterations

							//allocating memory undepandent of k
	sendCounts = (int*)malloc(numprocs * sizeof(int));
	assert(sendCounts != NULL);
	displsScatter = (int*)malloc(numprocs * sizeof(int));
	assert(displsScatter != NULL);
	recvCounts = (int*)malloc(numprocs * sizeof(int));
	assert(recvCounts != NULL);
	displsGather = (int*)malloc(numprocs * sizeof(int));
	assert(displsGather != NULL);
	dataFromFile = (int*)malloc(DATA_FROM_FILE_SIZE * sizeof(int));
	assert(dataFromFile != NULL);

	//Time measurment
	double t1 = MPI_Wtime();

	if (myid == 0)
	{
		
		//read vectors from data-set file
		vectorsReadFile = readVectorsFromFile(inputFile,
			numDims,
			&totalNumVectors,
			&k,
			&t,
			&dt,
			&iterationLimit,
			&requiredQuality,
			&pointsSpeeds);

		//vectorToClusterRelevance - the cluster id for each vector
		vectorToClusterRelevance = (int*)malloc(totalNumVectors * sizeof(int));
		assert(vectorToClusterRelevance != NULL);

		dataFromFile[0] = totalNumVectors;
		dataFromFile[1] = iterationLimit;
		dataFromFile[2] = k;
		dataFromFile[3] = t;
		
	}

	//broadcasting helpful data from file to all procs
	MPI_Bcast(dataFromFile, DATA_FROM_FILE_SIZE, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&requiredQuality, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&dt, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	totalNumVectors = dataFromFile[0];
	iterationLimit = dataFromFile[1];
	k = dataFromFile[2];
	t = dataFromFile[3];
	

	//compute the chunck of vectors each machine gets
	//set values to sendCounts & displsScatter
	createVectorAssignmentToMachinesArray(totalNumVectors, numprocs, numDims, sendCounts, displsScatter);

	//make arrangements to gather all the vector to cluster relevancies - for later on in the program 
	for (i = 0; i < numprocs; ++i) { recvCounts[i] = sendCounts[i] / numDims; }
	displsGather[0] = 0;
	for (i = 1; i < numprocs; ++i) { displsGather[i] = displsGather[i - 1] + recvCounts[i - 1]; }

	//sendCount[myid] is number of doubles every machine gets
	numVectorsInMachine = sendCounts[myid] / numDims;

	//allocate memory for storing vectors on each machine
	vectorsEachProc = (double**)malloc(numVectorsInMachine * sizeof(double*));
	assert(vectorsEachProc != NULL);
	vectorsEachProc[0] = (double*)malloc(numVectorsInMachine * numDims * sizeof(double));
	assert(vectorsEachProc[0] != NULL);
	for (i = 1; i < numVectorsInMachine; ++i)
	{
		vectorsEachProc[i] = vectorsEachProc[i - 1] + numDims;
	}
	pointsSpeedsEachProc = (double**)malloc(numVectorsInMachine * sizeof(double*));
	assert(pointsSpeedsEachProc != NULL);
	pointsSpeedsEachProc[0] = (double*)malloc(numVectorsInMachine * numDims * sizeof(double));
	for (i = 1; i < numVectorsInMachine; ++i)
	{
		pointsSpeedsEachProc[i] = pointsSpeedsEachProc[i - 1] + numDims;
	}

	//scatter vectors to all machines
	MPI_Scatterv(vectorsReadFile, sendCounts, displsScatter, MPI_DOUBLE, vectorsEachProc[0], sendCounts[myid], MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	//scatter point speeds to all machines
	MPI_Scatterv(pointsSpeeds, sendCounts, displsScatter, MPI_DOUBLE, pointsSpeedsEachProc[0], sendCounts[myid], MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	//each proc writes its vectors to the GPU for the k-Means
	copyVectorsToGPU(vectorsEachProc, &devVectors, pointsSpeedsEachProc, &devVectorSpeeds, numVectorsInMachine, numDims);
	
	//vectorToClusterRelevance - the cluster id for each vector
	vToCRelevanceEachProc = (int*)malloc(numVectorsInMachine * sizeof(int));
	assert(vToCRelevanceEachProc != NULL);

	//allocate memory for clusters matrix
	clusters = (double**)malloc(k * sizeof(double*));
	assert(clusters != NULL);
	clusters[0] = (double*)malloc(k * numDims * sizeof(double));
	assert(clusters[0] != NULL);
	for (i = 1; i < k; ++i)
	{
		clusters[i] = clusters[i - 1] + numDims;
	}

	//p0 picks first k elements in vectorsReadFile[] as initial cluster centers
	if (myid == 0)
	{
		for (i = 0; i < k; ++i)
		{
			for (j = 0; j < numDims; ++j)
			{
				clusters[i][j] = vectorsReadFile[j + i * numDims];
			}
		}
	}

	//p0 shares the initialized clusters with all procs
	MPI_Bcast(clusters[0], k * numDims, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	do
	{
		//save GPU run time - do not move the points with 0*vi
		if (currentT != 0)
		{
			
			//update the points on GPU
			movePointsWithCuda(vectorsEachProc, devVectors, devVectorSpeeds, numVectorsInMachine, numDims, dt);

			printPoints(vectorsEachProc[0], numVectorsInMachine, numDims);

			//update the points on CPU
			//movePointsWithOMP(vectorsEachProc, pointsSpeedsEachProc, numVectorsInMachine, numDims, dt);
		}

		/* start the core computation -------------------------------------------*/
		int check = k_means(vectorsEachProc, devVectors, numDims, numVectorsInMachine, k, iterationLimit, vToCRelevanceEachProc, clusters, MPI_COMM_WORLD);
		
		MPI_Gatherv(vToCRelevanceEachProc, numVectorsInMachine, MPI_INT, vectorToClusterRelevance,
			recvCounts, displsGather, MPI_INT, 0, MPI_COMM_WORLD);
		

		//computing cluster group quality
		if (myid == 0)
		{
			diameters = computeClustersDiameters(vectorsReadFile, totalNumVectors, k, numDims, vectorToClusterRelevance);

			clusterGroupQuality = computeClusterGroupQuality(clusters, k, numDims, diameters);

			free(diameters);
		}

		//broadcasting the found quality to all procs because it's a condition to stop the do..while loop
		MPI_Bcast(&clusterGroupQuality, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		currentT += dt;
		if (myid == 0)
		{
			printf("dt = %lf  qm = %lf\n", currentT, clusterGroupQuality);
			fflush(stdout);
		}
			
	} while (currentT < t && clusterGroupQuality > requiredQuality);

	
	if (myid == 0)
		writeClustersToFile(outputFile, clusters, k, numDims, clusterGroupQuality);
	
	//Time measurment
	double t2 = MPI_Wtime() - t1;

	//free all memory allocated
	FreeVectorsOnGPU(&devVectors);
	free(clusters[0]);
	free(clusters);
	free(vToCRelevanceEachProc);

	if (myid == 0)
	{
		free(vectorToClusterRelevance);
		printf("\ntime=%.5f\nquality=%.5f\n\n", t2, clusterGroupQuality);
	}

	MPI_Finalize();
}

void createVectorAssignmentToMachinesArray(int totalNumPoints,
	int  numOfMachines,
	int  numDims,
	int *sendCounts,
	int *displs)
{
	int i, remainder, index, *pointCounterForMachine;

	pointCounterForMachine = (int*)malloc(numOfMachines * sizeof(int));

	remainder = totalNumPoints % numOfMachines;
	index = 0;

	for (i = 0; i < numOfMachines; ++i)
	{
		pointCounterForMachine[i] = totalNumPoints / numOfMachines;
		if (remainder > 0)
		{
			pointCounterForMachine[i]++;
			remainder--;
		}

		sendCounts[i] = pointCounterForMachine[i] * numDims;
		displs[i] = index;
		index += sendCounts[i];
	}
}

void movePointsWithOMP(double **points, double **speeds, int numOfPoints, int numDims, double dt)
{
	int i, j;

#pragma parallel for private(j)
	for (i = 0; i < numOfPoints; i++)
	{
		for (j = 0; j < numDims; j++)
		{
			points[i][j] += speeds[i][j] * dt;
		}
	}
}

void printPoints(double *points, int numOfPoints, int numDims)
{
	int i, j;

	for  (i = 0; i < numOfPoints; i++)
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

