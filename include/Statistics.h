#define PATH "results/"

namespace RungeKuttaBenchmark{
  class Statistics{
    private:
      timing **_times;
      unsigned _times_count;
      unsigned _runs_count;

      float *getProcMeans();
      float *getMemoMeans();
      float *getProcStandardDeviations();
      float *getMemoStandardDeviations();
      float calculateProcMean(timing *t);
      float calculateMemoMean(timing *t);
      float calculateProcStandardDeviation(timing *t);
      float calculateMemoStandardDeviation(timing *t);
    public:
      Statistics();
      Statistics(unsigned runs_count);
      ~Statistics();
      void addTimingList(timing *t);
      void printHistograms(char *title);
      void printMeans(char *title);
      void printStandardDeviations(char *title);
  };
}