#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <xmmintrin.h>

#define THREADS 512

__global__ void GPUkernel(int N, float * x, float * y, float * z, float * m,
			  float * p, float * ax, float * ay, float * az, float eps2) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float pi = 0;
  float axi = 0;
  float ayi = 0;
  float azi = 0;
  float xi = x[i];
  float yi = y[i];
  float zi = z[i];
  __shared__ float xj[THREADS], yj[THREADS], zj[THREADS], mj[THREADS];
  for ( int jb=0; jb<N/blockDim.x; jb++ ) {
    __syncthreads();
    xj[threadIdx.x] = x[jb*blockDim.x+threadIdx.x];
    yj[threadIdx.x] = y[jb*blockDim.x+threadIdx.x];
    zj[threadIdx.x] = z[jb*blockDim.x+threadIdx.x];
    mj[threadIdx.x] = m[jb*blockDim.x+threadIdx.x];
    __syncthreads();
#pragma unroll
    for( int j=0; j<blockDim.x; j++ ) {
      float dx = xj[j] - xi;
      float dy = yj[j] - yi;
      float dz = zj[j] - zi;
      float R2 = dx * dx + dy * dy + dz * dz + eps2;
      float invR = rsqrtf(R2);
      pi += mj[j] * invR;
      float invR3 = mj[j] * invR * invR * invR;
      axi += dx * invR3;
      ayi += dy * invR3;
      azi += dz * invR3;
    }
  }
  p[i] = pi;
  ax[i] = axi;
  ay[i] = ayi;
  az[i] = azi;
}

int main() {
// Initialize
  int N = 1 << 16;
  float OPS = 20. * N * N * 1e-9;
  float EPS2 = 1e-6;
  int size = N * sizeof(float);
  int threads = THREADS;
  struct timeval tic, toc;
  float *x, *y, *z, *m, *p, *ax, *ay, *az;
  cudaMallocManaged(&x, size);
  cudaMallocManaged(&y, size);
  cudaMallocManaged(&z, size);
  cudaMallocManaged(&m, size);
  cudaMallocManaged(&p, size);
  cudaMallocManaged(&ax, size);
  cudaMallocManaged(&ay, size);
  cudaMallocManaged(&az, size);
  for (int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    z[i] = drand48();
    m[i] = drand48() / N;
  }
  printf("N      : %d\n",N);
// CUDA
  gettimeofday(&tic, NULL);
  GPUkernel<<<N/threads,threads>>>(N, x, y, z, m, p, ax, ay, az, EPS2);
  cudaDeviceSynchronize();
  gettimeofday(&toc, NULL);
  double time = toc.tv_sec-tic.tv_sec+(toc.tv_usec-tic.tv_usec)*1e-6;
  printf("CUDA   : %e s : %lf GFlops\n",time, OPS/time);

// No CUDA
  float pdiff = 0, pnorm = 0, adiff = 0, anorm = 0;
  gettimeofday(&tic, NULL);
#pragma omp parallel for reduction(+: pdiff, pnorm, adiff, anorm)
  for (int i=0; i<N; i++) {
    float pi = 0;
    float axi = 0;
    float ayi = 0;
    float azi = 0;
    float xi = x[i];
    float yi = y[i];
    float zi = z[i];
    for (int j=0; j<N; j++) {
      float dx = x[j] - xi;
      float dy = y[j] - yi;
      float dz = z[j] - zi;
      float R2 = dx * dx + dy * dy + dz * dz + EPS2;
      float invR = 1.0f / sqrtf(R2);
      float invR3 = m[j] * invR * invR * invR;
      pi += m[j] * invR;
      axi += dx * invR3;
      ayi += dy * invR3;
      azi += dz * invR3;
    }
    pdiff += (p[i] - pi) * (p[i] - pi);
    pnorm += pi * pi;
    adiff += (ax[i] - axi) * (ax[i] - axi)
      + (ay[i] - ayi) * (ay[i] - ayi)
      + (az[i] - azi) * (az[i] - azi);
    anorm += axi * axi + ayi * ayi + azi * azi;
  }
  gettimeofday(&toc, NULL);
  time = toc.tv_sec-tic.tv_sec+(toc.tv_usec-tic.tv_usec)*1e-6;
  printf("No CUDA: %e s : %lf GFlops\n",time, OPS/time);
  printf("P ERR  : %e\n",sqrt(pdiff/pnorm));
  printf("A ERR  : %e\n",sqrt(adiff/anorm));

// DEALLOCATE
  cudaFree(x);
  cudaFree(y);
  cudaFree(z);
  cudaFree(m);
  cudaFree(p);
  cudaFree(ax);
  cudaFree(ay);
  cudaFree(az);
  return 0;
}
