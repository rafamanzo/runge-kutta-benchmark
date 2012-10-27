#include <cstdlib>
#include <cstdio>

#include <rkb_types.h>
#include <DataSet.h>
#include <Fiber.h>
#include <Timer.h>
#include <CTimer.h>
#include <Fixture.h>
#include <CStraightFixture.h>
#include <Statistics.h>
#include <Benchmark.h>
#include <CBenchmark.h>
#include <RKCKernel.h>

using namespace RungeKuttaBenchmark;

timing *CBenchmark::runRK2(unsigned runs, unsigned initial_points){
  unsigned run, fiber_index;
  CTimer timer;
  Fiber **fibers;
  timing *t;
  CStraightFixture *cs;

  t = (timing *) malloc(runs*sizeof(timing));
  for(run = 0; run < runs; run++){
    t[run].proc = t[run].memo = 0.0;

    timer.startRecordMemoTime();
      cs = new CStraightFixture();
    timer.stopRecordMemoTime();

    timer.startRecordProcTime();
      rk2_caller((*cs).getInitialPoints(), initial_points, (*cs).getStepSize(), (*cs).getDataSet().n_x(), (*cs).getDataSet().n_y(),
                 (*cs).getDataSet().n_z(), (*cs).getDataSet().field(), &fibers
                );
    timer.stopRecordProcTime();
    t[run].memo += timer.getMemoTime();
    t[run].proc += timer.getProcTime();
    t[run].points_count = initial_points;
    delete cs;

    for(fiber_index = 0; fiber_index < initial_points; fiber_index++)
      delete fibers[fiber_index];

    free(fibers);

    timer.resetProcTime();
    timer.resetMemoTime();
  }

  return t;
}

timing *CBenchmark::runRK4(unsigned runs, unsigned initial_points){
  unsigned run, fiber_index;
  CTimer timer;
  Fiber **fibers;
  timing *t;
  CStraightFixture *cs;

  t = (timing *) malloc(runs*sizeof(timing));

  for(run = 0; run < runs; run++){
    t[run].proc = t[run].memo = 0.0;

    timer.startRecordMemoTime();
      cs = new CStraightFixture();
    timer.stopRecordMemoTime();

    timer.startRecordProcTime();
      rk4_caller((*cs).getInitialPoints(), initial_points, (*cs).getStepSize(), (*cs).getDataSet().n_x(), (*cs).getDataSet().n_y(),
                 (*cs).getDataSet().n_z(), (*cs).getDataSet().field(), &fibers
                );
    timer.stopRecordProcTime();
    t[run].memo += timer.getMemoTime();
    t[run].proc += timer.getProcTime();
    t[run].points_count = initial_points;
    delete cs;

    for(fiber_index = 0; fiber_index < initial_points; fiber_index++)
      delete fibers[fiber_index];

    free(fibers);

    timer.resetProcTime();
    timer.resetMemoTime();
  }

  return t;
}