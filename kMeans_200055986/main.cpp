#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "io.h"
#include "kMeans.h"

#define DATA_FROM_FILE_SIZE 2


void createVectorAssignmentToMachinesArray(int  totalNumVectors,
	int  numOfMachines,
	int  numDims,
	int *sendCounts,
	int *displs);

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
		t,							//maximum time for running the algorithm
		dt;							//the change in time in each iteration

	double	 *vectorsReadFile,		//[totalNumVectors] all vectors from data-set
		*devVectors = NULL,	//pointer to vectors on GPU 
		*diameters,			//[numClusters] in use by p0 only. contains diameters for each cluster
		**clusters,				//[k][numDims] contains cluster centers
		requiredQuality,		//required quality to reach stated in input file
		clusterGroupQuality,	//quality computed for each group of k clusters
		**vectorsEachProc;		//[numVectorsInMachine][numDims] holds all vectors that were assigned to a proc

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
			&k,
			&totalNumVectors,
			&iterationLimit,
			&requiredQuality);

		//vectorToClusterRelevance - the cluster id for each vector
		vectorToClusterRelevance = (int*)malloc(totalNumVectors * sizeof(int));
		assert(vectorToClusterRelevance != NULL);

		dataFromFile[0] = totalNumVectors;
		dataFromFile[1] = iterationLimit;
		
	}

	//broadcasting helpful data from file to all procs
	MPI_Bcast(dataFromFile, DATA_FROM_FILE_SIZE, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&requiredQuality, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	totalNumVectors = dataFromFile[0];
	iterationLimit = dataFromFile[1];

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

	//scatter vectors to all machines
	MPI_Scatterv(vectorsReadFile, sendCounts, displsScatter, MPI_DOUBLE, vectorsEachProc[0], sendCounts[myid], MPI_DOUBLE, 0, MPI_COMM_WORLD);
	printf("%d\n", myid);
	fflush(stdout);

	//each proc writes its vectors to the GPU for the k-Means
	copyVectorsToGPU(vectorsEachProc, &devVectors, numVectorsInMachine, numDims);

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

	MPI_Finalize();
}

void createVectorAssignmentToMachinesArray(int  totalNumVectors,
	int  numOfMachines,
	int  numDims,
	int *sendCounts,
	int *displs)
{
	int i, remainder, index, *vectorCounterForMachine;

	vectorCounterForMachine = (int*)malloc(numOfMachines * sizeof(int));

	remainder = totalNumVectors % numOfMachines;
	index = 0;

	for (i = 0; i < numOfMachines; ++i)
	{
		vectorCounterForMachine[i] = totalNumVectors / numOfMachines;
		if (remainder > 0)
		{
			vectorCounterForMachine[i]++;
			remainder--;
		}

		sendCounts[i] = vectorCounterForMachine[i] * numDims;
		displs[i] = index;
		index += sendCounts[i];
	}
}
