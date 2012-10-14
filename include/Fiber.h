namespace RungeKuttaBenchmark{
  class Fiber{
    private:
      unsigned _pointsCount;
      vector *_points;
      clock_t _allocation_clock_count;
    public:
      Fiber(unsigned pointsCount);
      Fiber();
      //~Fiber();
      void setPoint(unsigned order, vector point);
      vector getPoint(unsigned order);
      unsigned pointsCount();
      double getAllocationTime();
  };
}
