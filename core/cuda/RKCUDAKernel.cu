#include <cstdlib>
#include <cstdio>
#include <cmath>

#include "cuda.h"
#include "cuda_runtime.h"

#include <rkb_types.h>
#include <DataSet.h>
#include <Fiber.h>
#include <RKCUDAKernel.h>

/******************************/
/* Auxiliary Vector Functions */
/******************************/

/*FIXME: there must be libraries inside CUDA to work with vectors*/

__device__ int cuda_offset(int n_x, int n_y, int x, int y, int z){
  return x + n_x*y + n_y*n_x*z;
}

__device__ vector cuda_sum(vector v1, vector v2){
  vector sum;

  sum.x = v1.x + v2.x;
  sum.y = v1.y + v2.y;
  sum.z = v1.z + v2.z;

  return sum;
}

__device__ vector cuda_subtract(vector v1, vector v2){
  vector subtraction;

  subtraction.x = v1.x - v2.x;
  subtraction.y = v1.y - v2.y;
  subtraction.z = v1.z - v2.z;

  return subtraction;
}


__device__ vector cuda_mult_scalar(vector v, double scalar){
  vector mult;

  mult.x = v.x*scalar;
  mult.y = v.y*scalar;
  mult.z = v.z*scalar;

  return mult;
}

__device__ void cuda_set(vector *x, vector y){
  (*x).x = y.x;
  (*x).y = y.y;
  (*x).z = y.z;
}

__device__ double cuda_module(vector v){
  return sqrt(pow(v.x, 2) + pow(v.y, 2) + pow(v.z, 2));
}

__device__ double cuda_distance(vector x, vector y){
  return cuda_module(cuda_sum(x, cuda_mult_scalar(y, -1.0)));
}

/************************************/
/* Auxiliary Aproximation Functions */
/************************************/

__device__ vector cuda_nearest_neighbour(vector v0, int n_x, int n_y, int n_z, vector_field field){
  int x, y, z;
  vector zero;

  zero.x = zero.y = zero.z = 0.0;

  if( (v0.x - floor(v0.x)) > 0.5 && v0.x < (n_x - 1))
    x = (int) ceil(v0.x);
  else
    x = (int) floor(v0.x);

  if( (v0.y - floor(v0.y)) > 0.5 && v0.y < (n_y - 1))
    y = (int) ceil(v0.y);
  else
    y = (int) floor(v0.y);

  if( (v0.z - floor(v0.z)) > 0.5 && v0.z < (n_z - 1))
    z = (int) ceil(v0.z);
  else
    z = (int) floor(v0.z);

  if(x >= n_x || y >= n_y || z >= n_z || x < 0 || y < 0 || z < 0){
    return zero;
  }else{
    return field[cuda_offset(n_x, n_y, x, y, z)];
  }
}

__device__ vector cuda_trilinear_interpolation(vector v0, int n_x, int n_y, int n_z, vector_field field){
  int x1, y1, z1, x0, y0, z0;
  double xd, yd, zd;

  vector P1, P2, P3, P4, P5, P6, P7, P8, X1, X2, X3, X4, Y1, Y2, final;

  x1 = ceil(v0.x);
  y1 = ceil(v0.y);
  z1 = ceil(v0.z);
  x0 = floor(v0.x);
  y0 = floor(v0.y);
  z0 = floor(v0.z);
  xd = v0.x - x0;
  yd = v0.y - y0;
  zd = v0.z - z0;

  if(x1 >= n_x || y1 >= n_y || z1 >= n_z || x0 < 0 || y0 < 0 || z0 < 0){
    return cuda_nearest_neighbour(v0, n_x, n_y, n_z, field);
  }else{
    cuda_set(&P1, field[cuda_offset(n_x, n_y, x0, y0, z0)]);
    cuda_set(&P2, field[cuda_offset(n_x, n_y, x1, y0, z0)]);
    cuda_set(&P3, field[cuda_offset(n_x, n_y, x0, y0, z1)]);
    cuda_set(&P4, field[cuda_offset(n_x, n_y, x1, y0, z1)]);
    cuda_set(&P5, field[cuda_offset(n_x, n_y, x0, y1, z0)]);
    cuda_set(&P6, field[cuda_offset(n_x, n_y, x1, y1, z0)]);
    cuda_set(&P7, field[cuda_offset(n_x, n_y, x0, y1, z1)]);
    cuda_set(&P8, field[cuda_offset(n_x, n_y, x1, y1, z1)]);

    cuda_set(&X1, cuda_sum(P1, cuda_mult_scalar( cuda_subtract(P2, P1) , xd ) ));
    cuda_set(&X2, cuda_sum(P3, cuda_mult_scalar( cuda_subtract(P4, P3) , xd ) ));
    cuda_set(&X3, cuda_sum(P5, cuda_mult_scalar( cuda_subtract(P6, P5) , xd ) ));
    cuda_set(&X4, cuda_sum(P7, cuda_mult_scalar( cuda_subtract(P8, P7) , xd ) ));

    cuda_set(&Y1, cuda_sum(X1, cuda_mult_scalar( cuda_subtract(X3, X1) , yd ) ));
    cuda_set(&Y2, cuda_sum(X2, cuda_mult_scalar( cuda_subtract(X4, X2) , yd ) ));

    cuda_set(&final, cuda_sum(Y1, cuda_mult_scalar( cuda_subtract(Y2, Y1) , zd ) ));

    return final;
  }
}

/***********/
/* Kernels */
/***********/

__global__ void rk2_kernel(vector *v0, int count_v0, double h, int n_x, int n_y, int n_z, vector_field field, vector *points, int *n_points, int max_points){
  vector k1, k2, initial, direction;
  int i, n_points_aux;

  n_points_aux = 0;

  i = blockIdx.x * blockDim.x + threadIdx.x;

  cuda_set( &initial, v0[i] );
  cuda_set( &direction, field[cuda_offset(n_x, n_y, initial.x, initial.y, initial.z)] );

  while(cuda_module(direction) > 0.0 && (n_points_aux < max_points && n_points_aux < MAX_POINTS)){
    n_points_aux++;

    cuda_set( &(points[cuda_offset(count_v0, 0, i, n_points_aux - 1, 0)]), initial );

    cuda_set( &k1, cuda_mult_scalar( direction, h ) );
    cuda_set( &k2, cuda_mult_scalar( cuda_trilinear_interpolation(cuda_sum(initial, cuda_mult_scalar( k1, 0.5 )), n_x, n_y, n_z, field), h) );

    cuda_set( &initial, cuda_sum( initial, k2) );
    cuda_set( &direction, cuda_trilinear_interpolation(initial, n_x, n_y, n_z, field) );
  }

  n_points[i] = n_points_aux;
  n_points_aux = 0;
}

__global__ void rk4_kernel(vector *v0, int count_v0, double h, int n_x, int n_y, int n_z, vector_field field, vector *points, int *n_points, int max_points){
  vector k1, k2, k3, k4, initial, direction;
  int i, n_points_aux;

  n_points_aux = 0;

  i = blockIdx.x * blockDim.x + threadIdx.x;

  cuda_set( &initial, v0[i] );
  cuda_set( &direction, field[cuda_offset(n_x, n_y, initial.x, initial.y, initial.z)] );

  while(cuda_module(direction) > 0.0 && (n_points_aux < max_points && n_points_aux < MAX_POINTS)){
    n_points_aux++;

    cuda_set( &(points[cuda_offset(count_v0, 0, i, n_points_aux - 1, 0)]), initial );

    cuda_set( &k1, cuda_mult_scalar( direction, h ) );
    cuda_set( &k2, cuda_mult_scalar( cuda_trilinear_interpolation(cuda_sum(initial, cuda_mult_scalar( k1, 0.5 )), n_x, n_y, n_z, field), h) );
    cuda_set( &k3, cuda_mult_scalar( cuda_trilinear_interpolation(cuda_sum(initial, cuda_mult_scalar( k2, 0.5 )), n_x, n_y, n_z, field), h) );
    cuda_set( &k4, cuda_mult_scalar( cuda_trilinear_interpolation(cuda_sum(initial, k3), n_x, n_y, n_z, field), h) );

    cuda_set( &initial, cuda_sum( initial, cuda_sum( cuda_mult_scalar( k1 , 0.166666667 ), cuda_sum( cuda_mult_scalar( k2, 0.333333333 ), cuda_sum( cuda_mult_scalar( k3, 0.333333333 ), cuda_mult_scalar( k4, 0.166666667 ) ) ) ) ) );
    cuda_set( &direction, cuda_trilinear_interpolation(initial, n_x, n_y, n_z, field) );
  }

  n_points[i] = n_points_aux;
}

/***********/
/* Callers */
/***********/

int blockThreadLimit(int count_v0){
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  return ceil(((double) count_v0)/((double) deviceProp.maxThreadsPerBlock));
}

int blockRegisterLimit(int count_v0){
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  return ceil(((double) count_v0*REGISTERS_PER_THREAD) / ((double) deviceProp.regsPerBlock) );
}

int blocksCount(int count_v0){
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  if(deviceProp.multiProcessorCount > blockThreadLimit(count_v0) &&
     deviceProp.multiProcessorCount > blockRegisterLimit(count_v0)
    )
    return deviceProp.multiProcessorCount;
  else if(blockRegisterLimit(count_v0) > deviceProp.multiProcessorCount &&
          blockRegisterLimit(count_v0) > blockThreadLimit(count_v0)
         )
    return blockRegisterLimit(count_v0);
  else
    return blockThreadLimit(count_v0);
}

int threadsPerBlock(int count_v0){
  return floor( ((double) count_v0)/((double) blocksCount(count_v0)) );
}

void rk2_cuda_caller(vector *v0, int count_v0, double h, int n_x, int n_y, int n_z, vector_field field, vector *points, int *points_count, int max_points){
  rk2_kernel<<<blocksCount(count_v0), threadsPerBlock(count_v0)>>>(v0, count_v0, h, n_x, n_y, n_z, field, points, points_count, max_points);
  if(cudaDeviceSynchronize() != cudaSuccess){
    printf("There was an error on RK2 execution: %s\n", cudaGetErrorString(cudaGetLastError()));
    exit(-1);
  }
}

void rk4_cuda_caller(vector *v0, int count_v0, double h, int n_x, int n_y, int n_z, vector_field field, vector *points, int *points_count, int max_points){
  rk4_kernel<<<blocksCount(count_v0), threadsPerBlock(count_v0)>>>(v0, count_v0, h, n_x, n_y, n_z, field, points, points_count, max_points);
  if(cudaDeviceSynchronize() != cudaSuccess){
    printf("There was an error on RK4 execution: %s\n", cudaGetErrorString(cudaGetLastError()));
    exit(-1);
  }
}