namespace RungeKuttaBenchmark{
  class Benchmarker{
    private:
      void cudaRK2Benchmark(unsigned runs_count);
      void cudaRK4Benchmark(unsigned runs_count);
    public:
      void runCPUTests(unsigned runs_count);
      void runGPUTests(unsigned runs_count);
  };
}