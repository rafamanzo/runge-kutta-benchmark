#include <ctime>

#include <Timer.h>

using namespace RungeKuttaBenchmark;

Timer::Timer(){
  _proc_time_in_sec = 0.0;
  _memo_time_in_sec = 0.0;
}

float Timer::getProcTime(){
  return _proc_time_in_sec;
}

float Timer::getMemoTime(){
  return _memo_time_in_sec;
}

void Timer::resetProcTime(){
  _proc_time_in_sec = 0.0;
}

void Timer::resetMemoTime(){
  _memo_time_in_sec = 0.0;
}