#include <rkb_types.h>
#include <DataSet.h>
#include <Fixture.h>

using namespace RungeKuttaBenchmark;

DataSet Fixture::getDataSet(){
  return _data_set;
}

vector *Fixture::getInitialPoints(){
  return _v0;
}

float Fixture::getStepSize(){
  return _h;
}

unsigned Fixture::getInitialPointsCount(){
  return _v0_count;
}
