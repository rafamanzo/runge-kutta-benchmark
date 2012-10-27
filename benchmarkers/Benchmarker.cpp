#include <rkb_types.h>
#include <Statistics.h>
#include <Benchmark.h>
#include <CBenchmark.h>
#include <CUDABenchmark.h>
#include <Benchmarker.h>

using namespace RungeKuttaBenchmark;

void Benchmarker::runCPUTests(unsigned runs_count){
  CBenchmark *bm;
  Statistics *stats;

  bm = new CBenchmark();

  bm->run(runs_count, 16, 256, 2);

  stats = bm->getRK2Statistics();
  stats->printMeans((char *) "C-RK2");
  stats->printStandardDeviations((char *) "C-RK2");
  stats->printHistograms((char *) "C-RK2");

  stats = bm->getRK4Statistics();
  stats->printMeans((char *) "C-RK4");
  stats->printStandardDeviations((char *) "C-RK4");
  stats->printHistograms((char *) "C-RK4");

  delete bm;
}

void Benchmarker::runGPUTests(unsigned runs_count){
  CUDABenchmark *bm;
  Statistics *stats;

  bm = new CUDABenchmark();

  bm->run(runs_count, 16, 1024, 2);

  stats = bm->getRK2Statistics();
  stats->printMeans((char *) "CUDA-RK2");
  stats->printStandardDeviations((char *) "CUDA-RK2");
  stats->printHistograms((char *) "CUDA-RK2");

  stats = bm->getRK4Statistics();
  stats->printMeans((char *) "CUDA-RK4");
  stats->printStandardDeviations((char *) "CUDA-RK4");
  stats->printHistograms((char *) "CUDA-RK4");

  delete bm;
}