#include <rkb_types.h>
#include <Statistics.h>
#include <Benchmark.h>

using namespace RungeKuttaBenchmark;

Benchmark::~Benchmark(){
  delete _rk2_stats;
  delete _rk4_stats;
}

void Benchmark::run(unsigned runs, unsigned starting_initial_points_count, unsigned ending_initial_points_count, int step_size){
  unsigned initial_points_count;

  _rk2_stats = new Statistics(runs);
  _rk4_stats = new Statistics(runs);

  for(initial_points_count = starting_initial_points_count; initial_points_count <= ending_initial_points_count; initial_points_count *= step_size){
    _rk2_stats->addTimingList(runRK2(runs, initial_points_count));
    _rk4_stats->addTimingList(runRK4(runs, initial_points_count));
  }
}

Statistics *Benchmark::getRK2Statistics(){
  return _rk2_stats;
}

Statistics *Benchmark::getRK4Statistics(){
  return _rk4_stats;
}

