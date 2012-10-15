#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <ctime>

#include <DataSet.h>
#include <Fiber.h>
#include <RKCKernel.h>
#include <Timer.h>
#include <CTimer.h>
#include <Fixture.h>
#include <CStraightFixture.h>
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

  for(run = 0; run < runs_count; run++)
    memo_time[run] = proc_time[run] = 0.0;

  for(initial_points = 16; initial_points <= 256; initial_points *= 2){

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

    printf("\nRK2 Report for %d tests with %d initial points\n\nMedium performance:\n\tProccessing took: %fs\n\tMemory operations took: %fs\n\nStandard deviation for:\n\tProccessing: %f\n\tMemory operations: %f\n\n", runs_count, initial_points, proc_time_mean, memo_time_mean, sqrt(proc_time_variance), sqrt(memo_time_variance));
  }

  free(memo_time);
  free(proc_time);
}