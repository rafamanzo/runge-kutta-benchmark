cmake_minimum_required(VERSION 2.8)

PROJECT(RungeKuttaBenchmark)

find_package (Threads)

include_directories(include)

add_executable(RungeKuttaBenchmark Main core/DataSet fixtures/Fixture fixtures/cpp/CStraightFixture timers/Timer timers/CTimer core/Fiber Benchmarker core/cpp/RKCKernel)

target_link_libraries (RungeKuttaBenchmark ${CMAKE_THREAD_LIBS_INIT})