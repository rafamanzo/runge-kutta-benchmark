#include <Benchmarker.h>

using namespace RungeKuttaBenchmark;

int main(int argc, char const *argv[]){
  Benchmarker b;

  b.cppRK2Benchmark(30);

  return 0;
}