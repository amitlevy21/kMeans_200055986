#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "io.h"
#include "kMeans.h"
#include "kernel.h"
#include "quality.h"

#define DATA_FROM_FILE_INT_SIZE 4
#define DATA_FROM_FILE_DOUBLE_SIZE 2
#define P0 0


void createPointAssignmentToProcArray(int totalNumPoints,
	int  numOfProcs,
	int  numDims,
	int *sendCounts,	//[numOfProcs] item i will hold the num of elements proc i will receive in scatterv 
	int *displs);		//[numOfProcs] the offsets between the data of each proc


void pointCollectionAssigmentForGatherV(int* recvCounts, const int* sendCounts, int* displsGather,
	int* recvCountsUpdatedPoints, int* displsGatherUpdatedPoints, int numprocs, int numDims);

int main(int argc, char *argv[])
{
	//variables required for MPI initialization
	int numprocs, myid;

	//variables for program
	int i, j, k,					// k is actual num of clusters
		numDims = 2,
		totalNumPoints,				//num of points read from file
		iterationLimit,				//limit of iterations allowed in kMeans algorithm
		*pointToClusterRelevance,   //[totalNumPoints] in use by p0 only! index is point and value is cluster index which the point belongs to
		numPointsInProc,			//for each proc states how many points were assigned 
		*pToCRelevanceEachProc,		//[numPointsInProc] in use for the K-Means only! holds the relevance of points to clusters for each proc
		*sendCounts = NULL,			//[numprocs] used for scatterV of points to procs
		*recvCounts = NULL,			//[numprocs] used for gatherV of points to cluster relevancies from all procs to p0
		*recvCountsUpdatedPoints,	//[numprocs] used for gatherV of updated points of each proc to p0
		*displsScatter = NULL,		//[numprocs] used for scatterV of points to procs
		*displsGather = NULL,		//[numprocs] used for gatherV of points to cluster relevancies from all procs to p0
		*displsGatherUpdatedPoints, //[numprocs] used for gatherV of updated points of each proc to p0
		*dataFromFileInt,			//[DATA_FROM_FILE_INT_SIZE] data from file with type int that will be broadcasted later
		t;							//maximum time for running the algorithm

	double	 *pointsFromFile,		//[totalNumPoints * numDims] all points from data-set
		*devPoints = NULL,			//pointer to points on GPU 
		*devPointSpeeds = NULL,		//pointer to points speeds on GPU
		*diameters,					//[numClusters] in use by p0 only. contains diameters for each cluster
		**clusters,					//[k][numDims] contains cluster centers
		requiredQuality,			//required quality to reach stated in input file
		currentQuality,				//quality of the current state of points, before moving them again
		**pointsEachProc,			//[numPointsInProc][numDims] holds all points that were assigned to a proc
		*pointsSpeedsFromFile,		//[totalNumPoints * numDims] holds all speeds for each coordinate of each point
		**pointsSpeedsEachProc,		//[numPointsInProc][numDims] holds the speeds for the points that belong to proc
		*dataFromFileDouble,		//[DATA_FROM_FILE_DOUBLE_SIZE] data from file with type double that will be broadcasted later
		dt,							//the change in time in each iteration
		currentT = 0;				//the current time for each proc
		
	char    *inputFile = "C:\\data_kmeans.txt",
		*outputFile = "C:\\data_result.txt";

	//establishing MPI communication
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Status status;

	//allocating memory undepandent of k
	sendCounts = (int*)malloc(numprocs * sizeof(int));
	assert(sendCounts != NULL);
	displsScatter = (int*)malloc(numprocs * sizeof(int));
	assert(displsScatter != NULL);
	recvCounts = (int*)malloc(numprocs * sizeof(int));
	assert(recvCounts != NULL);
	displsGather = (int*)malloc(numprocs * sizeof(int));
	assert(displsGather != NULL);
	recvCountsUpdatedPoints = (int*)malloc(numprocs * sizeof(int));
	assert(recvCountsUpdatedPoints != NULL);
	displsGatherUpdatedPoints = (int*)malloc(numprocs * sizeof(int));
	assert(displsGatherUpdatedPoints != NULL);
	dataFromFileInt = (int*)malloc(DATA_FROM_FILE_INT_SIZE * sizeof(int));
	assert(dataFromFileInt != NULL);
	dataFromFileDouble = (double*)malloc(DATA_FROM_FILE_DOUBLE_SIZE * sizeof(double));
	assert(dataFromFileDouble != NULL);
	
	//Time measurment
	double t1 = MPI_Wtime();

	if (myid == 0)
	{
		
		//read points coordinate and speeds from data-set file
		pointsFromFile = readPointsDataFromFile(inputFile,
			numDims,
			&totalNumPoints,
			&k,
			&t,
			&dt,
			&iterationLimit,
			&requiredQuality,
			&pointsSpeedsFromFile);
		
		//pointToClusterRelevance - the cluster id for each point
		pointToClusterRelevance = (int*)malloc(totalNumPoints * sizeof(int));
		assert(pointToClusterRelevance != NULL);

		dataFromFileInt[0] = totalNumPoints;
		dataFromFileInt[1] = iterationLimit;
		dataFromFileInt[2] = k;
		dataFromFileInt[3] = t;

		dataFromFileDouble[0] = requiredQuality;
		dataFromFileDouble[1] = dt;
	}
	
	//broadcasting helpful data from file to all procs
	MPI_Bcast(dataFromFileInt, DATA_FROM_FILE_INT_SIZE, MPI_INT, P0, MPI_COMM_WORLD);
	MPI_Bcast(dataFromFileDouble, DATA_FROM_FILE_DOUBLE_SIZE, MPI_DOUBLE, P0, MPI_COMM_WORLD);
	
	totalNumPoints = dataFromFileInt[0];
	iterationLimit = dataFromFileInt[1];
	k = dataFromFileInt[2];
	t = dataFromFileInt[3];

	requiredQuality = dataFromFileDouble[0];
	dt = dataFromFileDouble[1];

	//compute the chunck of points each proc gets
	//set values to sendCounts & displsScatter
	createPointAssignmentToProcArray(totalNumPoints, numprocs, numDims, sendCounts, displsScatter);

	//make arrangements to gather all the point to cluster relevancies - for later on in the program
	pointCollectionAssigmentForGatherV(recvCounts, sendCounts, displsGather, recvCountsUpdatedPoints,
		displsGatherUpdatedPoints, numprocs, numDims);

	//sendCount[myid] is number of doubles every proc gets
	numPointsInProc = sendCounts[myid] / numDims;
	
	//allocate memory for storing points on each proc
	pointsEachProc = (double**)malloc(numPointsInProc * sizeof(double*));
	assert(pointsEachProc != NULL);
	pointsEachProc[0] = (double*)malloc(numPointsInProc * numDims * sizeof(double));
	assert(pointsEachProc[0] != NULL);
	for (i = 1; i < numPointsInProc; i++)
	{
		pointsEachProc[i] = pointsEachProc[i - 1] + numDims;
	}
	pointsSpeedsEachProc = (double**)malloc(numPointsInProc * sizeof(double*));
	assert(pointsSpeedsEachProc != NULL);
	pointsSpeedsEachProc[0] = (double*)malloc(numPointsInProc * numDims * sizeof(double));
	for (i = 1; i < numPointsInProc; i++)
	{
		pointsSpeedsEachProc[i] = pointsSpeedsEachProc[i - 1] + numDims;
	}

	//scatter points to all procs
	MPI_Scatterv(pointsFromFile, sendCounts, displsScatter, MPI_DOUBLE, pointsEachProc[0], sendCounts[myid], MPI_DOUBLE, P0, MPI_COMM_WORLD);
	
	//scatter point speeds to all procs
	MPI_Scatterv(pointsSpeedsFromFile, sendCounts, displsScatter, MPI_DOUBLE, pointsSpeedsEachProc[0], sendCounts[myid], MPI_DOUBLE, P0, MPI_COMM_WORLD);
	
	//each proc writes its points and their speeds to the GPU for the k-Means
	copyPointDataToGPU(pointsEachProc, &devPoints, pointsSpeedsEachProc, &devPointSpeeds, numPointsInProc, numDims);
	
	//pointToClusterRelevance - the cluster id for each point
	pToCRelevanceEachProc = (int*)malloc(numPointsInProc * sizeof(int));
	assert(pToCRelevanceEachProc != NULL);

	//allocate memory for clusters matrix
	clusters = (double**)malloc(k * sizeof(double*));
	assert(clusters != NULL);
	clusters[0] = (double*)malloc(k * numDims * sizeof(double));
	assert(clusters[0] != NULL);
	for (i = 1; i < k; i++)
	{
		clusters[i] = clusters[i - 1] + numDims;
	}

	//p0 picks first k elements in pointsReadFile[] as initial cluster centers
	if (myid == P0)
		pickFirstKAsInitialClusterCenters(clusters, k, pointsFromFile, totalNumPoints, numDims);

	//p0 shares the initialized clusters with all procs
	MPI_Bcast(clusters[0], k * numDims, MPI_DOUBLE, P0, MPI_COMM_WORLD);
	
	do // condition = while (currentT < t && currentQuality > requiredQuality)
	{
		//save GPU run time - do not move the points with 0*vi
		if (currentT != 0)
		{
			//update the points on GPU
			movePointsWithCuda(pointsEachProc, devPoints, devPointSpeeds, numPointsInProc, numDims, dt);

			//update the points on CPU
			//movePointsWithOMP(pointsEachProc, pointsSpeedsEachProc, numPointsInProc, numDims, dt);
		}
		
		/* start the core computation -------------------------------------------*/
		k_means(pointsEachProc, devPoints, numDims, numPointsInProc, k, iterationLimit, pToCRelevanceEachProc, clusters, MPI_COMM_WORLD);
		
		//Gather point to cluster relevance from all procs
		MPI_Gatherv(pToCRelevanceEachProc, numPointsInProc, MPI_INT, pointToClusterRelevance,
			recvCounts, displsGather, MPI_INT, P0, MPI_COMM_WORLD);
		//Gather moved points from all procs
		MPI_Gatherv(pointsEachProc[0], numPointsInProc, MPI_DOUBLE, pointsFromFile, recvCountsUpdatedPoints,
			displsGatherUpdatedPoints, MPI_DOUBLE, P0, MPI_COMM_WORLD);
		

		//computing cluster group quality
		if (myid == P0)
		{
			diameters = computeClustersDiameters(pointsFromFile, totalNumPoints, k, numDims, pointToClusterRelevance);
			
			currentQuality = computeClusterGroupQuality(clusters, k, numDims, diameters);
			
			free(diameters);
		}

		//broadcasting the found quality to all procs because it's a condition to stop the do..while loop
		MPI_Bcast(&currentQuality, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		if (myid == P0)
		{
			printf("dt = %.2lf  qm = %.2lf\n", currentT, currentQuality);
			fflush(stdout);
		}

		currentT += dt;
			
	} while (currentT < t && currentQuality > requiredQuality);

	if (myid == P0)
		writeClustersToFile(outputFile, clusters, k, numDims, currentQuality);
	
	//Time measurment
	double t2 = MPI_Wtime() - t1;

	//free all memory allocated
	FreePointDataOnGPU(&devPoints, &devPointSpeeds);
	free(clusters[0]);
	free(clusters);
	free(pToCRelevanceEachProc);
	free(pointsEachProc[0]);
	free(pointsEachProc);
	free(sendCounts);
	free(displsScatter);
	free(recvCounts);
	free(recvCountsUpdatedPoints);
	free(displsGatherUpdatedPoints);
	free(displsGather);
	free(dataFromFileInt);
	free(dataFromFileDouble);

	if (myid == P0)
	{
		free(pointsFromFile);
		free(pointsSpeedsFromFile);
		free(pointToClusterRelevance);
		printf("\ntime=%.5f\nquality=%.5f\n\n", t2, currentQuality);
	}

	MPI_Finalize();
}

//for load balancing
void createPointAssignmentToProcArray(int totalNumPoints,
	int  numOfProcs,
	int  numDims,
	int *sendCounts,	//[numOfProcs] item i will hold the num of elements proc i will receive in scatterv 
	int *displs)		//[numOfProcs] the offsets between the data of each proc
{
	int i, remainder, index, *pointCounterForProc;

	pointCounterForProc = (int*)malloc(numOfProcs * sizeof(int));

	remainder = totalNumPoints % numOfProcs;
	index = 0;

	for (i = 0; i < numOfProcs; i++)
	{
		pointCounterForProc[i] = totalNumPoints / numOfProcs;
		if (remainder > 0)
		{
			pointCounterForProc[i]++;
			remainder--;
		}

		sendCounts[i] = pointCounterForProc[i] * numDims;
		displs[i] = index;
		index += sendCounts[i];
	}
	free(pointCounterForProc);
}

void pointCollectionAssigmentForGatherV(int* recvCounts, const int* sendCounts, int* displsGather,
	int* recvCountsUpdatedPoints, int* displsGatherUpdatedPoints, int numprocs, int numDims)
{
	int i, j;

	//1st section - handles values to recieve the point to cluster relevance from each proc
	for (i = 0; i < numprocs; i++) { recvCounts[i] = sendCounts[i] / numDims; }
	displsGather[0] = 0;
	for (i = 1; i < numprocs; i++) { displsGather[i] = displsGather[i - 1] + recvCounts[i - 1]; }

	//2nd section - handles values to recieve the moved points from each proc
	for (i = 0; i < numprocs; i++) { recvCountsUpdatedPoints[i] = sendCounts[i]; }
	displsGatherUpdatedPoints[0] = 0;
	for (i = 1; i < numprocs; i++) { displsGatherUpdatedPoints[i] = displsGatherUpdatedPoints[i - 1] + recvCountsUpdatedPoints[i - 1]; }

}