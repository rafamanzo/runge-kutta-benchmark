namespace RungeKuttaBenchmark{
  class CTimer : public Timer{
    public:
      virtual void startRecordProcTime();
      virtual void stopRecordProcTime();
      virtual void startRecordMemoTime();
      virtual void stopRecordMemoTime();
    private:
      clock_t _proc_start;
      clock_t _memo_start;
  };
}