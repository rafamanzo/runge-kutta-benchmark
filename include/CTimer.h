namespace RungeKuttaBenchmark{
  class CTimer : public Timer{
    public:
      virtual void startRecordProcTime();
      virtual void stopRecordProcTime();
      virtual void startRecordMemoTime();
      virtual void stopRecordMemoTime();
  };
}