#ifndef __QUALITY_H_
#define __QUALITY_H_

#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <assert.h>


double euclidDistanceForQuality(int    dim,  	//no. dimensions 
	double *v1,   	//[numdims] 
	double *v2);   	//[numdims]

double* computeClustersDiameters(double *vectors,	//complete set of vectors from file
	int    numVectors,
	int    numClusters,//some k
	int    numDims,
	int    *vToCR);	//[numVectors] contains cluster relevancy

double computeClusterGroupQuality(double **clusters, //[numClusters][numDims] contains all cluster centers
	int numClusters,
	int numDims,
	double *diameters);//[numClusters] holds all diameters of clusters

#endif // !__QUALITY_H_
