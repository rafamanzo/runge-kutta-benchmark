namespace RungeKuttaBenchmark{
  class CUDATimer : public Timer{
    public:
      CUDATimer();
      ~CUDATimer();
      virtual void startRecordProcTime();
      virtual void stopRecordProcTime();
      virtual void startRecordMemoTime();
      virtual void stopRecordMemoTime();
    private:
      cudaEvent_t _proc_start;
      cudaEvent_t _proc_finish;
      cudaEvent_t _memo_start;
      cudaEvent_t _memo_finish;
  };
}