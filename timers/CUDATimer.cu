#include "cuda.h"
#include "cuda_runtime.h"

#include <Timer.h>
#include <CUDATimer.h>

using namespace RungeKuttaBenchmark;

CUDATimer::CUDATimer(){
  cudaEventCreate(&_proc_start);
  cudaEventCreate(&_proc_finish);
  cudaEventCreate(&_memo_start);
  cudaEventCreate(&_memo_finish);
}

CUDATimer::~CUDATimer(){
  cudaEventDestroy(_proc_start);
  cudaEventDestroy(_proc_finish);
  cudaEventDestroy(_memo_start);
  cudaEventDestroy(_memo_finish);
}

void CUDATimer::startRecordProcTime(){
  cudaEventRecord(_proc_start, 0);
}

void CUDATimer::stopRecordProcTime(){
  float time;

  cudaEventRecord(_proc_finish, 0);
  cudaEventSynchronize(_proc_finish);

  cudaEventElapsedTime(&time, _proc_start, _proc_finish);

  _proc_time_in_sec += (double) (time/1000.0);
}

void CUDATimer::startRecordMemoTime(){
  cudaEventRecord(_memo_start, 0);
}

void CUDATimer::stopRecordMemoTime(){
  float time;

  cudaEventRecord(_memo_finish, 0);
  cudaEventSynchronize(_memo_finish);

  cudaEventElapsedTime(&time, _memo_start, _memo_finish);

  _memo_time_in_sec += (double) (time/1000.0);
}