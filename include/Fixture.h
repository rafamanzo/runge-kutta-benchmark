#define X_MAX_SIZE 512
#define Y_MAX_SIZE 256
#define Z_MAX_SIZE 256

namespace RungeKuttaBenchmark{
  class Fixture{
    public:
      DataSet getDataSet();
      vector *getInitialPoints();
      unsigned getInitialPointsCount();
      float getStepSize();
    protected:
      DataSet _data_set;
      unsigned _v0_count;
      vector *_v0;
      float _h;
  };
}