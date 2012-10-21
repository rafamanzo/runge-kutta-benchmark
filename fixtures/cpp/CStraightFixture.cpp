#include <cstdlib>
#include <cstdio>

#include <DataSet.h>
#include <Fixture.h>
#include <CStraightFixture.h>

using namespace RungeKuttaBenchmark;

CStraightFixture::CStraightFixture(){
  unsigned i, j, k;
  vector_field field;
  vector direction;

  direction.y = 1.0;
  direction.x = direction.z = 0.0;

  _v0 = NULL;
  _v0_count = X_MAX_SIZE;

  _h = 0.01;

  field = (vector *) malloc((X_MAX_SIZE*Y_MAX_SIZE*Z_MAX_SIZE)*sizeof(vector));
  if(field == NULL)
    printf("Could not allocate the data for the benchmark\n");

  _v0 = (vector *) malloc((X_MAX_SIZE)*sizeof(vector));
  if(_v0 == NULL)
    printf("Could not allocate the data for the benchmark\n");

  for (i = 0; i < X_MAX_SIZE; i++)
    for (j = 0; j < Y_MAX_SIZE; j++)
      for (k = 0; k < Z_MAX_SIZE; k++)
        field[DataSet::offset(X_MAX_SIZE, Y_MAX_SIZE, i, j, k)] = direction;

  for (i = 0; i < X_MAX_SIZE; ++i){
    _v0[i].x = i;
    _v0[i].y = 0;
    _v0[i].z = Z_MAX_SIZE/2;
  }

  _data_set = DataSet(X_MAX_SIZE, Y_MAX_SIZE, Z_MAX_SIZE, field);
}

CStraightFixture::~CStraightFixture(){
  free(_data_set.field());
  free(_v0);
}