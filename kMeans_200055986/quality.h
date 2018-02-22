#ifndef __QUALITY_H_
#define __QUALITY_H_

#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <assert.h>

double euclidDistanceForQuality(int numDims,  	//no. dimensions 
	double *p1,   		//[numDims] 
	double *p2);   		//[numDims] 

double* computeClustersDiameters(double *points,	//complete set of points from file
	int    numPoints,
	int    numClusters,		
	int    numDims,
	int    *pToCR);			//[numPoints] contains cluster relevancy

double computeClusterGroupQuality(double **clusters, //[numClusters][numDims] contains all cluster centers
	int numClusters,
	int numDims,
	double *diameters);		//[numClusters] holds all diameters of clusters

#endif // !__QUALITY_H_
