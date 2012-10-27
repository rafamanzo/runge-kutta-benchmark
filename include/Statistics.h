#define PATH "results/"

namespace RungeKuttaBenchmark{
  class Statistics{
    private:
      timing **_times;
      unsigned _times_count;
      unsigned _runs_count;

      double *getProcMeans();
      double *getMemoMeans();
      double *getProcStandardDeviations();
      double *getMemoStandardDeviations();
      double calculateProcMean(timing *t);
      double calculateMemoMean(timing *t);
      double calculateProcStandardDeviation(timing *t);
      double calculateMemoStandardDeviation(timing *t);
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