cmake_minimum_required(VERSION 2.8)

PROJECT(RungeKuttaBenchmark)

find_package (Threads)
find_package (CUDA)

include_directories(include)
include(FindCUDA)

#-Xptxas=-v for profilling
#-G -g for debug
#-arch=sm_21 is an alias for -arch=compute_21 -code=compute_21
set( my_cuda_flags "-arch=sm_21" CACHE STRING "Set the cuda flags like -arch=sm_21")
list(APPEND CUDA_NVCC_FLAGS ${my_cuda_flags})

cuda_add_executable(RungeKuttaBenchmark Main
                                        core/DataSet core/Fiber core/cpp/RKCKernel core/cuda/RKCUDAKernel.cu
                                        fixtures/Fixture fixtures/cpp/CStraightFixture fixtures/cuda/CUDAStraightFixture.cu
                                        timers/Timer timers/CTimer timers/CUDATimer.cu
                                        benchmarkers/Benchmarker benchmarkers/Benchmark benchmarkers/CBenchmark benchmarkers/Statistics benchmarkers/CUDABenchmark)

target_link_libraries (RungeKuttaBenchmark ${CMAKE_THREAD_LIBS_INIT})