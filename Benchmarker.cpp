#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <ctime>

#include "cuda.h"
#include "cuda_runtime.h"

#include <DataSet.h>
#include <Fiber.h>
#include <RKCKernel.h>
#include <RKCUDAKernel.h>
#include <Timer.h>
#include <CTimer.h>
#include <CUDATimer.h>
#include <Fixture.h>
#include <CStraightFixture.h>
#include <CUDAStraightFixture.h>
#include <Benchmarker.h>

using namespace RungeKuttaBenchmark;

void Benchmarker::cppRK2Benchmark(unsigned runs_count){
  unsigned run, fiber_index;
  int initial_points;
  CTimer timer;
  Fiber **fibers;
  double *proc_time, *memo_time, proc_time_variance, proc_time_mean, memo_time_variance, memo_time_mean;
  CStraightFixture *cs;

  memo_time = (double *) malloc(runs_count*sizeof(double));
  proc_time = (double *) malloc(runs_count*sizeof(double));


  for(initial_points = 16; initial_points <= 256; initial_points *= 2){
    for(run = 0; run < runs_count; run++)
      memo_time[run] = proc_time[run] = 0.0;

    for(run = 0; run < runs_count; run++){
      timer.startRecordMemoTime();
      cs = new CStraightFixture();
      timer.stopRecordMemoTime();

      timer.startRecordProcTime();
      rk2_caller((*cs).getInitialPoints(), initial_points, (*cs).getStepSize(), (*cs).getDataSet().n_x(), (*cs).getDataSet().n_y(), (*cs).getDataSet().n_z(), (*cs).getDataSet().field(), &fibers);
      timer.stopRecordProcTime();
      memo_time[run] += timer.getMemoTime();
      proc_time[run] += timer.getProcTime();
      delete cs;

      for(fiber_index = 0; fiber_index < initial_points; fiber_index++)
        delete fibers[fiber_index];

      free(fibers);

      timer.resetProcTime();
      timer.resetMemoTime();
    }

    proc_time_mean = memo_time_mean = 0;
    for(run = 0; run < runs_count; run++){
      proc_time_mean += proc_time[run];
      memo_time_mean += memo_time[run];
    }
    proc_time_mean /= runs_count;
    memo_time_mean /= runs_count;

    proc_time_variance = memo_time_variance = 0.0;
    for(run = 0; run < runs_count; run++){
      proc_time_variance += pow((proc_time[run] - proc_time_mean), 2);
      memo_time_variance += pow((memo_time[run] - memo_time_mean), 2);
    }
    proc_time_variance /= runs_count;
    memo_time_variance /= runs_count;

    printf("\nMultithreaded C++ RK2 Report for %d tests with %d initial points\n\nMedium performance:\n\tProccessing took: %fs\n\tMemory operations took: %fs\n\nStandard deviation for:\n\tProccessing: %f\n\tMemory operations: %f\n\n", runs_count, initial_points, proc_time_mean, memo_time_mean, sqrt(proc_time_variance), sqrt(memo_time_variance));
  }

  free(memo_time);
  free(proc_time);
}

void Benchmarker::cppRK4Benchmark(unsigned runs_count){
  unsigned run, fiber_index;
  int initial_points;
  CTimer timer;
  Fiber **fibers;
  double *proc_time, *memo_time, proc_time_variance, proc_time_mean, memo_time_variance, memo_time_mean;
  CStraightFixture *cs;

  memo_time = (double *) malloc(runs_count*sizeof(double));
  proc_time = (double *) malloc(runs_count*sizeof(double));

  for(initial_points = 16; initial_points <= 256; initial_points *= 2){
    for(run = 0; run < runs_count; run++)
      memo_time[run] = proc_time[run] = 0.0;

    for(run = 0; run < runs_count; run++){
      timer.startRecordMemoTime();
      cs = new CStraightFixture();
      timer.stopRecordMemoTime();

      timer.startRecordProcTime();
      rk4_caller((*cs).getInitialPoints(), initial_points, (*cs).getStepSize(), (*cs).getDataSet().n_x(), (*cs).getDataSet().n_y(), (*cs).getDataSet().n_z(), (*cs).getDataSet().field(), &fibers);
      timer.stopRecordProcTime();
      memo_time[run] += timer.getMemoTime();
      proc_time[run] += timer.getProcTime();
      delete cs;

      for(fiber_index = 0; fiber_index < initial_points; fiber_index++){
        delete fibers[fiber_index];
      }

      free(fibers);

      timer.resetProcTime();
      timer.resetMemoTime();
    }

    proc_time_mean = memo_time_mean = 0;
    for(run = 0; run < runs_count; run++){
      proc_time_mean += proc_time[run];
      memo_time_mean += memo_time[run];
    }
    proc_time_mean /= runs_count;
    memo_time_mean /= runs_count;

    proc_time_variance = memo_time_variance = 0.0;
    for(run = 0; run < runs_count; run++){
      proc_time_variance += pow((proc_time[run] - proc_time_mean), 2);
      memo_time_variance += pow((memo_time[run] - memo_time_mean), 2);
    }
    proc_time_variance /= runs_count;
    memo_time_variance /= runs_count;

    printf("\nMultithreaded C++ RK4 Report for %d tests with %d initial points\n\nMedium performance:\n\tProccessing took: %fs\n\tMemory operations took: %fs\n\nStandard deviation for:\n\tProccessing: %f\n\tMemory operations: %f\n\n", runs_count, initial_points, proc_time_mean, memo_time_mean, sqrt(proc_time_variance), sqrt(memo_time_variance));
  }

  free(memo_time);
  free(proc_time);
}

void Benchmarker::cudaRK2Benchmark(unsigned runs_count){
  CUDATimer cuda_timer;
  CTimer c_timer;
  CUDAStraightFixture *cds;
  CStraightFixture *cs;
  Fiber **fibers;
  int *points_count_aux;
  int i, j, initial_points;
  vector *points_aux;
  double *proc_time, *memo_time, proc_time_variance, proc_time_mean, memo_time_variance, memo_time_mean;
  unsigned run;

  memo_time = (double *) malloc(runs_count*sizeof(double));
  proc_time = (double *) malloc(runs_count*sizeof(double));

  for(initial_points = 16; initial_points <= 256; initial_points *= 2){
    for(run = 0; run < runs_count; run++)
      memo_time[run] = proc_time[run] = 0.0;

    for(run = 0; run < runs_count; run++){
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
        cudaMemcpy(points_count_aux, cds->getPointsCount(), initial_points*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(points_aux, cds->getPoints(), initial_points*cds->getMaxPoints()*sizeof(vector), cudaMemcpyDeviceToHost);
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

      memo_time[run] += c_timer.getMemoTime() + cuda_timer.getMemoTime();
      proc_time[run] += c_timer.getProcTime() + cuda_timer.getProcTime();

      c_timer.resetMemoTime();
      c_timer.resetProcTime();
      cuda_timer.resetMemoTime();
      cuda_timer.resetProcTime();
    }

    proc_time_mean = memo_time_mean = 0;
    for(run = 0; run < runs_count; run++){
      proc_time_mean += proc_time[run];
      memo_time_mean += memo_time[run];
    }
    proc_time_mean /= runs_count;
    memo_time_mean /= runs_count;

    proc_time_variance = memo_time_variance = 0.0;
    for(run = 0; run < runs_count; run++){
      proc_time_variance += pow((proc_time[run] - proc_time_mean), 2);
      memo_time_variance += pow((memo_time[run] - memo_time_mean), 2);
    }
    proc_time_variance /= runs_count;
    memo_time_variance /= runs_count;

    printf("\nCUDA RK2 Report for %d tests with %d initial points\n\nMedium performance:\n\tProccessing took: %fs\n\tMemory operations took: %fs\n\nStandard deviation for:\n\tProccessing: %f\n\tMemory operations: %f\n\n", runs_count, initial_points, proc_time_mean, memo_time_mean, sqrt(proc_time_variance), sqrt(memo_time_variance));
  }

  free(memo_time);
  free(proc_time);
}

void Benchmarker::cudaRK4Benchmark(unsigned runs_count){
  CUDATimer cuda_timer;
  CTimer c_timer;
  CUDAStraightFixture *cds;
  CStraightFixture *cs;
  Fiber **fibers;
  int *points_count_aux;
  int i, j, initial_points;
  vector *points_aux;
  double *proc_time, *memo_time, proc_time_variance, proc_time_mean, memo_time_variance, memo_time_mean;
  unsigned run;

  memo_time = (double *) malloc(runs_count*sizeof(double));
  proc_time = (double *) malloc(runs_count*sizeof(double));

  for(initial_points = 16; initial_points <= 256; initial_points *= 2){
    for(run = 0; run < runs_count; run++)
      memo_time[run] = proc_time[run] = 0.0;

    for(run = 0; run < runs_count; run++){
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
        cudaMemcpy(points_count_aux, cds->getPointsCount(), initial_points*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(points_aux, cds->getPoints(), initial_points*cds->getMaxPoints()*sizeof(vector), cudaMemcpyDeviceToHost);
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

      memo_time[run] += c_timer.getMemoTime() + cuda_timer.getMemoTime();
      proc_time[run] += c_timer.getProcTime() + cuda_timer.getProcTime();

      c_timer.resetMemoTime();
      c_timer.resetProcTime();
      cuda_timer.resetMemoTime();
      cuda_timer.resetProcTime();
    }

    proc_time_mean = memo_time_mean = 0;
    for(run = 0; run < runs_count; run++){
      proc_time_mean += proc_time[run];
      memo_time_mean += memo_time[run];
    }
    proc_time_mean /= runs_count;
    memo_time_mean /= runs_count;

    proc_time_variance = memo_time_variance = 0.0;
    for(run = 0; run < runs_count; run++){
      proc_time_variance += pow((proc_time[run] - proc_time_mean), 2);
      memo_time_variance += pow((memo_time[run] - memo_time_mean), 2);
    }
    proc_time_variance /= runs_count;
    memo_time_variance /= runs_count;

    printf("\nCUDA RK4 Report for %d tests with %d initial points\n\nMedium performance:\n\tProccessing took: %fs\n\tMemory operations took: %fs\n\nStandard deviation for:\n\tProccessing: %f\n\tMemory operations: %f\n\n", runs_count, initial_points, proc_time_mean, memo_time_mean, sqrt(proc_time_variance), sqrt(memo_time_variance));
  }

  free(memo_time);
  free(proc_time);
}

void Benchmarker::runCPUTests(unsigned runs_count){
  cppRK4Benchmark(runs_count);
  cppRK2Benchmark(runs_count);
}

void Benchmarker::runGPUTests(unsigned runs_count){
  cudaRK4Benchmark(runs_count);
  cudaRK2Benchmark(runs_count);
}