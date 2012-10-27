#include <cstdlib>
#include <cmath>
#include <cstdio>

#include <rkb_types.h>
#include <Statistics.h>

using namespace RungeKuttaBenchmark;

Statistics::Statistics(){
  _times = NULL;
  _times_count = 0;
  _runs_count = 0;
}

Statistics::Statistics(unsigned runs_count){
  _times = NULL;
  _times_count = 0;
  _runs_count = runs_count;
}

Statistics::~Statistics(){
  unsigned time_index;

  if(_times != NULL){
    for(time_index = 0; time_index < _times_count; time_index++)
      if(_times[time_index] != NULL){
        free(_times[time_index]);
        _times[time_index] = NULL;
      }

    free(_times);
    _times = NULL;
    _times_count = 0;
  }
}

void Statistics::addTimingList(timing *t){
  _times_count++;
  _times = (timing **) realloc(_times, _times_count*sizeof(timing *));
  _times[_times_count - 1] = t;
}

void Statistics::printHistograms(char *title){
  unsigned time_index, timing_index;
  FILE *proc_histogram, *memo_histogram;
  char file_name[257];

  for(time_index = 0; time_index < _times_count; time_index++){
    sprintf(file_name, "%sproccessing-histogram-%d-%s.dat", PATH, _times[time_index][0].points_count, title);
    proc_histogram = fopen(file_name, "w");

    sprintf(file_name, "%smemory-histogram-%d-%s.dat", PATH, _times[time_index][0].points_count, title);
    memo_histogram = fopen(file_name, "w");

    for(timing_index = 0; timing_index < _runs_count; timing_index++){
      fprintf(proc_histogram, "%f\n", _times[time_index][timing_index].proc);
      fprintf(memo_histogram, "%f\n", _times[time_index][timing_index].memo);
    }

    fclose(proc_histogram);
    fclose(memo_histogram);
  }
}

void Statistics::printMeans(char *title){
  unsigned time_index;
  double *proc_means, *memo_means;
  FILE *proc_means_data, *memo_means_data;
  char file_name[257];

  sprintf(file_name, "%sproccessing-means-%s.dat", PATH, title);
  proc_means_data = fopen(file_name, "w");
  sprintf(file_name, "%smemory-means-%s.dat", PATH, title);
  memo_means_data = fopen(file_name, "w");

  proc_means = getProcMeans();
  memo_means = getMemoMeans();
  for(time_index = 0; time_index < _times_count; time_index++){
    fprintf(proc_means_data, "%d %f\n", _times[time_index][0].points_count, proc_means[time_index]);
    fprintf(memo_means_data, "%d %f\n", _times[time_index][0].points_count, memo_means[time_index]);
  }

  free(proc_means);
  free(memo_means);

  fclose(proc_means_data);
  fclose(memo_means_data);
}

void Statistics::printStandardDeviations(char *title){
  unsigned time_index;
  double *proc_standard_deviations, *memo_standard_deviations;
  FILE *proc_standard_deviations_data, *memo_standard_deviations_data;
  char file_name[257];

  sprintf(file_name, "%sproccessing-standarddeviations-%s.dat", PATH, title);
  proc_standard_deviations_data = fopen(file_name, "w");
  sprintf(file_name, "%smemory-standarddeviations-%s.dat", PATH, title);
  memo_standard_deviations_data = fopen(file_name, "w");

  proc_standard_deviations = getProcStandardDeviations();
  memo_standard_deviations = getMemoStandardDeviations();
  for(time_index = 0; time_index < _times_count; time_index++){
    fprintf(proc_standard_deviations_data, "%d %f\n", _times[time_index][0].points_count, proc_standard_deviations[time_index]);
    fprintf(memo_standard_deviations_data, "%d %f\n", _times[time_index][0].points_count, memo_standard_deviations[time_index]);
  }

  free(proc_standard_deviations);
  free(memo_standard_deviations);

  fclose(proc_standard_deviations_data);
  fclose(memo_standard_deviations_data);
}

double *Statistics::getProcMeans(){
  double *means;
  unsigned time_index;

  means = (double *) malloc(_times_count*sizeof(double));

  for(time_index = 0; time_index < _times_count; time_index++)
    means[time_index] = calculateProcMean(_times[time_index]);

  return means;
}

double *Statistics::getMemoMeans(){
  double *means;
  unsigned time_index;

  means = (double *) malloc(_times_count*sizeof(double));

  for(time_index = 0; time_index < _times_count; time_index++)
    means[time_index] = calculateMemoMean(_times[time_index]);

  return means;
}

double *Statistics::getProcStandardDeviations(){
  double *deviations;
  unsigned time_index;

  deviations = (double *) malloc(_times_count*sizeof(double));

  for(time_index = 0; time_index < _times_count; time_index++)
    deviations[time_index] = calculateProcStandardDeviation(_times[time_index]);

  return deviations;
}

double *Statistics::getMemoStandardDeviations(){
  double *deviations;
  unsigned time_index;

  deviations = (double *) malloc(_times_count*sizeof(double));

  for(time_index = 0; time_index < _times_count; time_index++)
    deviations[time_index] = calculateProcStandardDeviation(_times[time_index]);

  return deviations;
}

double Statistics::calculateProcMean(timing *t){
  unsigned timing_index;
  double sum;

  for(timing_index = 0; timing_index < _runs_count; timing_index++)
    sum += t[timing_index].proc;

  return sum/( (double) _runs_count);
}

double Statistics::calculateMemoMean(timing *t){
  unsigned timing_index;
  double sum;

  for(timing_index = 0; timing_index < _runs_count; timing_index++)
    sum += t[timing_index].memo;

  return sum/( (double) _runs_count);
}

double Statistics::calculateProcStandardDeviation(timing *t){
  unsigned timing_index;
  double mean, sum;

  mean = calculateProcMean(t);

  for(timing_index = 0; timing_index < _runs_count; timing_index++)
    sum += pow((t[timing_index].proc - mean), 2);

  return sqrt(sum/( (double) _runs_count));
}

double Statistics::calculateMemoStandardDeviation(timing *t){
  unsigned timing_index;
  double mean, sum;

  mean = calculateMemoMean(t);

  for(timing_index = 0; timing_index < _runs_count; timing_index++)
    sum += pow((t[timing_index].memo - mean), 2);

  return sqrt(sum/( (double) _runs_count));
}