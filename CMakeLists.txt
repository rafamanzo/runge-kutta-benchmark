cmake_minimum_required(VERSION 2.8)

PROJECT(RungeKuttaBenchmark)

find_package (Threads)
find_package (CUDA)

include_directories(include)
include(FindCUDA)

set( my_cuda_flags "-arch=sm_20" CACHE STRING "Set the cuda flags like -arch=sm_20")
list(APPEND CUDA_NVCC_FLAGS ${my_cuda_flags})

cuda_add_executable(RungeKuttaBenchmark Main core/DataSet fixtures/Fixture fixtures/cpp/CStraightFixture timers/Timer timers/CTimer core/Fiber Benchmarker core/cpp/RKCKernel core/cuda/RKCUDAKernel.cu timers/CUDATimer.cu fixtures/cuda/CUDAStraightFixture.cu)

target_link_libraries (RungeKuttaBenchmark ${CMAKE_THREAD_LIBS_INIT})