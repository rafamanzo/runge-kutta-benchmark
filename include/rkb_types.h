typedef struct vec{
  double x;
  double y;
  double z;
} vector;

typedef vector *vector_field;

typedef struct tmg{
  double proc;
  double memo;
  unsigned points_count;
} timing;