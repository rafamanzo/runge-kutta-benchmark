namespace RungeKuttaBenchmark{
  class Timer{
    public:
      Timer();
      virtual void startRecordProcTime() = 0;
      virtual void stopRecordProcTime() = 0;
      virtual void startRecordMemoTime() = 0;
      virtual void stopRecordMemoTime() = 0;
      double getProcTime();
      double getMemoTime();
      void resetProcTime();
      void resetMemoTime();
    protected:
      double _proc_time_in_sec;
      double _memo_time_in_sec;
      clock_t _proc_start;
      clock_t _memo_start;
  };
}