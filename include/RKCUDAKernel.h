#define MAX_POINTS 10000
#define REGISTERS_PER_THREAD 52 //Every device has an limit of registers per block that we must respect. Every time the kernel is modified this constant must be update acording to the result of compiling with the flag -Xptxas=-v

extern "C" void rk2_cuda_caller(vector *v0, int count_v0, double h, int n_x, int n_y, int n_z, vector_field field, vector *points, int *points_count, int max_points);
extern "C" void rk4_cuda_caller(vector *v0, int count_v0, double h, int n_x, int n_y, int n_z, vector_field field, vector *points, int *points_count, int max_points);