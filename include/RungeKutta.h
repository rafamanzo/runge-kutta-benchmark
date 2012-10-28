namespace RungeKuttaBenchmark{
  class RungeKutta{
    private:
      DataSet _dataset;
      vector *_v0;
      unsigned _count_v0;
      float _h;
    public:
      RungeKutta(DataSet dataset, vector *v0, unsigned count_v0, float h); 
      Fiber *order2();
      Fiber *order4();
  };
}
