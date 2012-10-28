#include <ctime>

#include <Timer.h>
#include <CTimer.h>

using namespace RungeKuttaBenchmark;

void CTimer::startRecordProcTime(){
  _proc_start = clock();
}

void CTimer::stopRecordProcTime(){
  _proc_time_in_sec += ( (clock() - _proc_start)/((float) CLOCKS_PER_SEC));
}

void CTimer::startRecordMemoTime(){
  _memo_start = clock();
}

void CTimer::stopRecordMemoTime(){
  _memo_time_in_sec += ( (clock() - _memo_start)/((float) CLOCKS_PER_SEC));
}