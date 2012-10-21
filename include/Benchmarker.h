namespace RungeKuttaBenchmark{
  class Benchmarker{
    private:
      void cppRK2Benchmark(unsigned runs_count);
      void cppRK4Benchmark(unsigned runs_count);
      void cudaRK2Benchmark(unsigned runs_count);
      void cudaRK4Benchmark(unsigned runs_count);
    public:
      void runCPUTests(unsigned runs_count);
      void runGPUTests(unsigned runs_count);
  };
}