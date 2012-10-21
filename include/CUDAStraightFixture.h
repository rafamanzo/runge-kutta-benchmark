namespace RungeKuttaBenchmark{
  class CUDAStraightFixture : public Fixture{
    private:
      vector_field _field;
      vector *_points;
      int *_points_count;
      int _max_points;
    public:
      CUDAStraightFixture(vector *v0, int v0_count, DataSet dataset);
      ~CUDAStraightFixture();
      vector *getPoints();
      int *getPointsCount();
      int getMaxPoints();
      vector_field getField();
  };
}