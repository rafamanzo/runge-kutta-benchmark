#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <DataSet.h>
#include <Fiber.h>

using namespace RungeKuttaBenchmark;

Fiber::Fiber(unsigned pointsCount){
  clock_t start;

  start = clock();

  _pointsCount = pointsCount;
  _points = (vector *) malloc(pointsCount*sizeof(vector));
  _allocation_clock_count = (clock() - start);
}

Fiber::Fiber(){
  clock_t start;

  start = clock();

  _pointsCount = 0;
  _points = NULL;

  _allocation_clock_count = (clock() - start);
}

Fiber::~Fiber(){
  if(_points != NULL){
    _pointsCount = 0;
    free(_points);
    _points = NULL;
  }
}

void Fiber::setPoint(unsigned order, vector point){
  clock_t start;

  start = clock();
  _points[order] = point;

  _allocation_clock_count += (clock() - start);
}

double Fiber::getAllocationTime(){
  return ( ( (double) _allocation_clock_count ) / ( (double) CLOCKS_PER_SEC ) );
}

vector Fiber::getPoint(unsigned order){
  return _points[order];
}
unsigned Fiber::pointsCount(){
  return _pointsCount;
}
