#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <sys/time.h>

#define M 8 //
#define K 8 


/*
for (int mb = 0; mb < M; mb += Mtile)
for (int nb = 0; nb < N; nb += Ntile)
for (int kb = 0; kb < K; kb += Ktile)
{
// compute Mtile-by-Ntile-by-Ktile matrix product
for (int k = 0; k < Ktile; ++k)
for (int i = 0; i < Mtile; ++i)
for (int j = 0; j < Ntile; ++j)
{
int row = mb + i;
int col = nb + j;
C[row][col] +=
A[row][kb + k] * B[kb + k][col];
}
}

*/

__global__ void matmul(float *A, float *B, float *C, int N) {
	int ii = threadIdx.x + blockDim.x * blockIdx.x;
	int jj = threadIdx.y + blockDim.y * blockIdx.y;
	for (int i=ii; i<ii+M; i++) {
		for (int j=jj; j<jj+M; j++) {
			float sum = 0.0f;
			for (int k=0; k<N; k++) {
				sum += A[N*i+k] * B[N*k+j];
			}
			C[N*i+j] = sum;
		}
	}
}

int main(int argc, char **argv) {
	int N = atoi(argv[1]);


	// Allocate memory space for matrices to cpu (host)
	float * h_A = new float [N*N]; // First matrix 
	float * h_B = new float [N*N]; // Second matrix
	float * h_C = new float [N*N]; // Result matrix

	// Allocate memory space for matrices to gpu (device)
	float *d_A, *d_B, *d_C; // Gpu allocations
	int size = N * N * sizeof(float); // Byte size for cuda malloc
	cudaMalloc((void **) &d_A, size); 
	cudaMalloc((void **) &d_B, size);
	cudaMalloc((void **) &d_C, size);

	// Init cpu matrices with random values.
	for (int i=0; i<N; i++) {
		for (int j=0; j<N; j++) {
			h_A[N*i+j] = drand48();
			h_B[N*i+j] = drand48();
			h_C[N*i+j] = 0;
		}
	}


	// Copy matrices to gpu memory
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);


	// Make gpu multiplication
	struct timeval tic, toc;
	gettimeofday(&tic, NULL); //Start timer

	// Invoke gpu multiplication
	dim3 grid(N/M/K, N/M/K); // Amount of 2 dimensional blocks per axis
	dim3 block(K,K); // Thread size per axis
	matmul<<<grid,block>>>(d_A, d_B, d_C, N);  
	cudaDeviceSynchronize();

	// Calculate and print flops
	gettimeofday(&toc, NULL); //End timer
	double time = toc.tv_sec-tic.tv_sec+(toc.tv_usec-tic.tv_usec)*1e-6;
	printf("N=%d: %lf s (%lf GFlops)\n",N,time,2.*N*N*N/time/1e9); // Print flops for gpu multiplication


	// Copy matrices back to cpu memory
	cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);


	// Make controll by trivially recalculating matrix values and subtracting them from the current cpu output matrix (and time)
	gettimeofday(&tic, NULL); //Start timer
	#pragma omp parallel for
	for (int i=0; i<N; i++) {
		for (int k=0; k<N; k++) {
			for (int j=0; j<N; j++) {
				h_C[N*i+j] -= h_A[N*i+k] * h_B[N*k+j];
			}
		}
	}
	gettimeofday(&toc, NULL); //End timer
	time = toc.tv_sec-tic.tv_sec+(toc.tv_usec-tic.tv_usec)*1e-6;
	printf("N=%d: %lf s (%lf GFlops)\n",N,time,2.*N*N*N/time/1e9); // Print flops for cpu multiplication

	// Total error: Sum difference on each value between cpu and gpu calculation
	float err = 0;
	for (int i=0; i<N; i++) {
		for (int j=0; j<N; j++) {
			err += fabs(h_C[N*i+j]);
		}
	}
	printf("error: %f\n",err/N/N); // Print total error

	// Clear memory
	delete[] h_A;
	delete[] h_B;
	delete[] h_C;
 	cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}
