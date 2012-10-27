namespace RungeKuttaBenchmark{
  class CUDABenchmark : public Benchmark{
    public:
      virtual timing *runRK2(unsigned runs, unsigned initial_points);
      virtual timing *runRK4(unsigned runs, unsigned initial_points);
    };
}