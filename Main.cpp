#include <Benchmarker.h>

using namespace RungeKuttaBenchmark;

int main(int argc, char const *argv[]){
  Benchmarker b;

  //b.runCPUTests(30);
  b.runGPUTests(30);

  return 0;
}