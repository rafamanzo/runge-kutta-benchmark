#include <cstdlib>
#include <cstdio>

#include "cuda.h"
#include "cuda_runtime.h"

#include <rkb_types.h>
#include <DataSet.h>
#include <Fiber.h>
#include <Timer.h>
#include <CTimer.h>
#include <CUDATimer.h>
#include <Fixture.h>
#include <CStraightFixture.h>
#include <CUDAStraightFixture.h>
#include <Statistics.h>
#include <Benchmark.h>
#include <CUDABenchmark.h>
#include <RKCUDAKernel.h>

using namespace RungeKuttaBenchmark;

timing *CUDABenchmark::runRK2(unsigned runs, unsigned initial_points){
  unsigned run, fiber_index;
  CUDATimer cuda_timer;
  CTimer c_timer;
  Fiber **fibers;
  timing *t;
  CUDAStraightFixture *cds;
  CStraightFixture *cs;
  int *points_count_aux;
  int i, j;
  vector *points_aux;

  t = (timing *) malloc(runs*sizeof(timing));
  for(run = 0; run < runs; run++){
    t[run].proc = t[run].memo = 0.0;

    c_timer.startRecordMemoTime();
      cs = new CStraightFixture();
    c_timer.stopRecordMemoTime();
    cuda_timer.startRecordMemoTime();
      cds = new CUDAStraightFixture(cs->getInitialPoints(), cs->getInitialPointsCount(), cs->getDataSet());
    cuda_timer.stopRecordMemoTime();

    cuda_timer.startRecordProcTime();
      rk2_cuda_caller(cds->getInitialPoints(), initial_points, cds->getStepSize(), cds->getDataSet().n_x(), cds->getDataSet().n_y(), cds->getDataSet().n_z(), cds->getField(), cds->getPoints(), cds->getPointsCount(), cds->getMaxPoints());
    cuda_timer.stopRecordProcTime();

    c_timer.startRecordMemoTime();
      points_count_aux = (int *) malloc(initial_points*sizeof(int));
      points_aux = (vector *) malloc(initial_points*cds->getMaxPoints()*sizeof(vector));
    c_timer.stopRecordMemoTime();

    cuda_timer.startRecordMemoTime();
      if(cudaMemcpy(points_count_aux, cds->getPointsCount(), initial_points*sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess){
        printf("\nCould not retrieve %d points count that should be at %p. Message: %s\n",initial_points, cds->getPointsCount(), cudaGetErrorString(cudaGetLastError()));
        exit(-1);
      }
      if(cudaMemcpy(points_aux, cds->getPoints(), initial_points*cds->getMaxPoints()*sizeof(vector), cudaMemcpyDeviceToHost) != cudaSuccess){
        printf("\nCould not retrieve %d lists of points that should be at %p\n",initial_points, cds->getPoints());
        exit(-1);
      }
    cuda_timer.stopRecordMemoTime();

    c_timer.startRecordMemoTime();
      fibers = (Fiber **) malloc(initial_points*sizeof(Fiber *));
      for(i = 0; i < initial_points; i++){
        fibers[i] = new Fiber(points_count_aux[i]);
        for(j = 0; j < points_count_aux[i]; j++){
          fibers[i]->setPoint(j, points_aux[DataSet::offset(initial_points, 0, i, j, 0)]);
        }
      }
    c_timer.stopRecordMemoTime();

    for(i = 0; i < initial_points; ++i)
      delete fibers[i];

    free(points_aux);
    free(points_count_aux);
    free(fibers);

    delete cds;
    delete cs;

    t[run].memo += c_timer.getMemoTime() + cuda_timer.getMemoTime();
    t[run].proc += c_timer.getProcTime() + cuda_timer.getProcTime();
    t[run].points_count = initial_points;

    c_timer.resetMemoTime();
    c_timer.resetProcTime();
    cuda_timer.resetMemoTime();
    cuda_timer.resetProcTime();
  }

  return t;
}

timing *CUDABenchmark::runRK4(unsigned runs, unsigned initial_points){
  unsigned run, fiber_index;
  CUDATimer cuda_timer;
  CTimer c_timer;
  Fiber **fibers;
  timing *t;
  CUDAStraightFixture *cds;
  CStraightFixture *cs;
  int *points_count_aux;
  int i, j;
  vector *points_aux;

  t = (timing *) malloc(runs*sizeof(timing));
  for(run = 0; run < runs; run++){
    t[run].proc = t[run].memo = 0.0;

    c_timer.startRecordMemoTime();
      cs = new CStraightFixture();
    c_timer.stopRecordMemoTime();
    cuda_timer.startRecordMemoTime();
      cds = new CUDAStraightFixture(cs->getInitialPoints(), cs->getInitialPointsCount(), cs->getDataSet());
    cuda_timer.stopRecordMemoTime();

    cuda_timer.startRecordProcTime();
      rk4_cuda_caller(cds->getInitialPoints(), initial_points, cds->getStepSize(), cds->getDataSet().n_x(), cds->getDataSet().n_y(), cds->getDataSet().n_z(), cds->getField(), cds->getPoints(), cds->getPointsCount(), cds->getMaxPoints());
    cuda_timer.stopRecordProcTime();

    c_timer.startRecordMemoTime();
      points_count_aux = (int *) malloc(initial_points*sizeof(int));
      points_aux = (vector *) malloc(initial_points*cds->getMaxPoints()*sizeof(vector));
    c_timer.stopRecordMemoTime();

    cuda_timer.startRecordMemoTime();
      if(cudaMemcpy(points_count_aux, cds->getPointsCount(), initial_points*sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess){
        printf("\nCould not retrieve %d points count that should be at %p. Message: %s\n",initial_points, cds->getPointsCount(), cudaGetErrorString(cudaGetLastError()));
        exit(-1);
      }
      if(cudaMemcpy(points_aux, cds->getPoints(), initial_points*cds->getMaxPoints()*sizeof(vector), cudaMemcpyDeviceToHost) != cudaSuccess){
        printf("\nCould not retrieve %d lists of points that should be at %p\n",initial_points, cds->getPoints());
        exit(-1);
      }
    cuda_timer.stopRecordMemoTime();

    c_timer.startRecordMemoTime();
      fibers = (Fiber **) malloc(initial_points*sizeof(Fiber *));
      for(i = 0; i < initial_points; i++){
        fibers[i] = new Fiber(points_count_aux[i]);
        for(j = 0; j < points_count_aux[i]; j++){
          fibers[i]->setPoint(j, points_aux[DataSet::offset(initial_points, 0, i, j, 0)]);
        }
      }
    c_timer.stopRecordMemoTime();

    for(i = 0; i < initial_points; ++i)
      delete fibers[i];

    free(points_aux);
    free(points_count_aux);
    free(fibers);

    delete cds;
    delete cs;

    t[run].memo += c_timer.getMemoTime() + cuda_timer.getMemoTime();
    t[run].proc += c_timer.getProcTime() + cuda_timer.getProcTime();
    t[run].points_count = initial_points;

    c_timer.resetMemoTime();
    c_timer.resetProcTime();
    cuda_timer.resetMemoTime();
    cuda_timer.resetProcTime();
  }

  return t;
}