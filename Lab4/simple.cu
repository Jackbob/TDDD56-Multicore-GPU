// Simple CUDA example by Ingemar Ragnemalm 2009. Simplest possible?
// Assigns every element in an array with its index.

// nvcc simple.cu -L /usr/local/cuda/lib -lcudart -o simple

#include <stdio.h>

const int N = 16; 
const int blocksize = 16; 

__global__ 
void threadnumber(float *c) 
{
	c[threadIdx.x] = threadIdx.x;
}

__global__ 
void squareroot(float *c) 
{
	c[threadIdx.x] = sqrt(c[threadIdx.x]);
}

__global__ 
void add_matrix(float *a, float *b, float *result)
{
	int row = (blockIdx.y * blockDim.y + threadIdx.y);
	int col = (blockIdx.x * blockDim.x + threadIdx.x);
	int e_per_row = (gridDim.x*blockDim.x);
	int idx = row*e_per_row + col;
	result[idx] = a[idx] + b[idx];
}

int main()
{
	/* First part, 1D */
	float *c = new float[N];
	float *cd;
	const int size = N*sizeof(float);
	for(int i=0; i<N; ++i){
		c[i] = 25.0f;
	}
	
	cudaMalloc( (void**)&cd, size );
	dim3 dimBlock1d( blocksize, 1 );
	dim3 dimGrid( 1, 1 );
	cudaMemcpy( cd, c, size, cudaMemcpyHostToDevice );
	squareroot<<<dimGrid, dimBlock1d>>>(cd);
	cudaThreadSynchronize();
	cudaMemcpy( c, cd, size, cudaMemcpyDeviceToHost ); 
	cudaFree( cd );

	printf("Square root: \n");
	for (int i = 0; i < N; i++)
		printf("%f ", c[i]);
	printf("\n");


	/* Second part, matrix */
	float *gpu_result, *gpu_ma, *gpu_mb;
	float* result = new float[N*N];
	float *ma = new float[N*N];
	float *mb = new float[N*N];
	const int matrixsize = N*N*sizeof(float);
	for(int i=0; i<N*N; ++i){
		ma[i] = (float)i;
		mb[i] = (float)i;
	}

	cudaMalloc( (void**)&gpu_result, matrixsize );
	cudaMalloc( (void**)&gpu_ma, matrixsize );
	cudaMalloc( (void**)&gpu_mb, matrixsize );
	dim3 dimBlockmatrix( 8, 8 );
	dim3 dimGridmatrix( 2, 2 );
	cudaMemcpy( gpu_ma, ma, matrixsize, cudaMemcpyHostToDevice );
	cudaMemcpy( gpu_mb, mb, matrixsize, cudaMemcpyHostToDevice );
	add_matrix<<<dimGridmatrix, dimBlockmatrix>>>(gpu_ma, gpu_mb, gpu_result);
	cudaThreadSynchronize();
	cudaMemcpy( result, gpu_result, matrixsize, cudaMemcpyDeviceToHost ); 
	cudaFree( gpu_ma );
	cudaFree( gpu_mb );
	cudaFree( gpu_result );

	printf("Matrix addition: \n");
	for (int r = 0; r < N; r++){
		for (int c = 0; c < N; c++)
			printf("%f ", result[r*N + c]);

		printf("\n");
	}
	

	printf("done\n");
	return EXIT_SUCCESS;
}
