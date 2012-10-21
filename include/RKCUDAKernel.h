#define MAX_POINTS 10000

extern "C" void rk2_cuda_caller(vector *v0, int count_v0, double h, int n_x, int n_y, int n_z, vector_field field, vector *points, int *points_count, int max_points);
extern "C" void rk4_cuda_caller(vector *v0, int count_v0, double h, int n_x, int n_y, int n_z, vector_field field, vector *points, int *points_count, int max_points);