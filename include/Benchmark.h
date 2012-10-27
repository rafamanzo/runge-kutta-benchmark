namespace RungeKuttaBenchmark{
  class Benchmark{
    private:
      Statistics *_rk2_stats;
      Statistics *_rk4_stats;
    protected:
      virtual timing *runRK2(unsigned runs, unsigned initial_points) = 0;
      virtual timing *runRK4(unsigned runs, unsigned initial_points) = 0;
    public:
      ~Benchmark();
      void run(unsigned runs, unsigned starting_initial_points_count, unsigned ending_initial_points_count, int step_size);
      Statistics *getRK2Statistics();
      Statistics *getRK4Statistics();
  };
}