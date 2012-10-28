namespace RungeKuttaBenchmark{
  class Timer{
    public:
      Timer();
      virtual void startRecordProcTime() = 0;
      virtual void stopRecordProcTime() = 0;
      virtual void startRecordMemoTime() = 0;
      virtual void stopRecordMemoTime() = 0;
      float getProcTime();
      float getMemoTime();
      void resetProcTime();
      void resetMemoTime();
    protected:
      float _proc_time_in_sec;
      float _memo_time_in_sec;
  };
}