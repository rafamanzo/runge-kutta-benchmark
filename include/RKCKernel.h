#define MAX_POINTS 10000

typedef struct ker_args{
  int id;
  vector *v0;
  int count_v0;
  float h;
  int n_x;
  int n_y;
  int n_z;
  vector_field field;
  RungeKuttaBenchmark::Fiber **fibers;
} kernel_args;

void rk2_caller(vector *v0, int count_v0, float h, int n_x, int n_y, int n_z, vector_field field, RungeKuttaBenchmark::Fiber ***fibers);
void rk4_caller(vector *v0, int count_v0, float h, int n_x, int n_y, int n_z, vector_field field, RungeKuttaBenchmark::Fiber ***fibers);