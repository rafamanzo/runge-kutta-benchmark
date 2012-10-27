#include <cstdlib>
#include <cstdio>

#include "cuda.h"
#include "cuda_runtime.h"

#include <rkb_types.h>
#include <DataSet.h>
#include <Fiber.h>
#include <Fixture.h>
#include <CUDAStraightFixture.h>

using namespace RungeKuttaBenchmark;

CUDAStraightFixture::CUDAStraightFixture(vector *v0, int v0_count, DataSet dataset){
  size_t available, total;

  _v0_count = v0_count;
  _data_set = dataset;

  if(cudaMalloc(&_v0, _v0_count*sizeof(vector)) == cudaErrorMemoryAllocation){
    printf("\nCould not allocate %fMB for the initial points\n", (_v0_count*sizeof(vector))/1024.0/1024.0);
    exit(-1);
  }
  if(cudaMalloc(&_field, dataset.n_x()*dataset.n_y()*dataset.n_z()*sizeof(vector)) == cudaErrorMemoryAllocation){
    printf("\nCould not allocate %fMB for the vector field\n", (dataset.n_x()*dataset.n_y()*dataset.n_z()*sizeof(vector))/1024.0/1024.0);
    exit(-1);
  }
  if(cudaMalloc(&_points_count, _v0_count*sizeof(int)) == cudaErrorMemoryAllocation){
    printf("\nCould not allocate %fMB for the points count vector\n", (_v0_count*sizeof(vector))/1024.0/1024.0);
    exit(-1);
  }
  cudaMemGetInfo(&available, &total);
  _max_points = ((available*0.9)/(sizeof(vector)*_v0_count));
  if(cudaMalloc(&_points, _v0_count*_max_points*sizeof(vector)) == cudaErrorMemoryAllocation){
    printf("\nCould not allocate %fMB for the fibers\n", (_v0_count*_max_points*sizeof(vector))/1024.0/1024.0);
    exit(-1);
  }

  if(cudaMemcpy(_v0, v0, _v0_count*sizeof(vector), cudaMemcpyHostToDevice) != cudaSuccess){
    printf("Failed to transfer the initial points list to device\n");
    exit(-1);
  }

  if(cudaMemcpy(_field, dataset.field(), dataset.n_x()*dataset.n_y()*dataset.n_z()*sizeof(vector), cudaMemcpyHostToDevice) != cudaSuccess){
    printf("Failed to transfer the vector field to the device\n");
    exit(-1);
  }
}

CUDAStraightFixture::~CUDAStraightFixture(){
  cudaFree(_field);
  cudaFree(_v0);
  cudaFree(_points);
  cudaFree(_points_count);
}

vector *CUDAStraightFixture::getPoints(){
  return _points;
}

int *CUDAStraightFixture::getPointsCount(){
  return _points_count;
}

int CUDAStraightFixture::getMaxPoints(){
  return _max_points;
}

vector_field CUDAStraightFixture::getField(){
  return _field;
}