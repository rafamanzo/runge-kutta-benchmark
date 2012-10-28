typedef struct vec{
  float x;
  float y;
  float z;
} vector;

typedef vector *vector_field;

typedef struct tmg{
  float proc;
  float memo;
  unsigned points_count;
} timing;