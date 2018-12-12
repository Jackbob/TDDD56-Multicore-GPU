// Simple CUDA example by Ingemar Ragnemalm 2009. Simplest possible?
// Assigns every element in an array with its index.

// nvcc simple.cu -L /usr/local/cuda/lib -lcudart -o simple

#include <stdio.h>
#include <chrono>

const int N = 16384; 
const int blocksize1d = 64; 

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
	int idx = col*e_per_row + row;
	result[idx] = a[idx] + b[idx];
}

void cpu_add_matrix(float *a, float *b, float *result){
    for (int i = 0; i < N*N; ++i)
		result[i] = a[i] + b[i];
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
	dim3 dimBlock1d( N, 1 );
	dim3 dimGrid( 1, 1 );
	cudaMemcpy( cd, c, size, cudaMemcpyHostToDevice );
	squareroot<<<dimGrid, dimBlock1d>>>(cd);
	cudaThreadSynchronize();
	cudaMemcpy( c, cd, size, cudaMemcpyDeviceToHost ); 

	/*printf("Square root: \n");
	for (int i = 0; i < N; i++)
		printf("%f ", c[i]);
	printf("\n");
    */

	/* Second part, matrix and cuda event*/
	float *gpu_result, *gpu_ma, *gpu_mb;
	float* result = new float[N*N];
    float* cpuresult = new float[N*N];
	float *ma = new float[N*N];
	float *mb = new float[N*N];
	const int matrixsize = N*N*sizeof(float);
	for(int i=0; i<N*N; ++i){
		ma[i] = (float)i;
		mb[i] = (float)i;
	}

    cudaEvent_t startEvent, endEvent;

    //Allocate and send matrices to GPU
	cudaMalloc( (void**)&gpu_result, matrixsize );
	cudaMalloc( (void**)&gpu_ma, matrixsize );
	cudaMalloc( (void**)&gpu_mb, matrixsize );
	dim3 dimBlockmatrix( blocksize1d, blocksize1d );
	dim3 dimGridmatrix( N/blocksize1d, N/blocksize1d );
	cudaMemcpy( gpu_ma, ma, matrixsize, cudaMemcpyHostToDevice );
	cudaMemcpy( gpu_mb, mb, matrixsize, cudaMemcpyHostToDevice );

    //Time start
    cudaEventCreate(&startEvent);
    cudaEventRecord(startEvent, 0);
    cudaEventSynchronize(startEvent);

    //Computation
	add_matrix<<<dimGridmatrix, dimBlockmatrix>>>(gpu_ma, gpu_mb, gpu_result);   
	cudaThreadSynchronize();

    //Time end
    cudaEventCreate(&endEvent);
    cudaEventRecord(endEvent, 0);
    cudaEventSynchronize(endEvent); 

    //Get result from GPU
	cudaMemcpy( result, gpu_result, matrixsize, cudaMemcpyDeviceToHost ); 
    
    float gpuTime = 0.0f;
    cudaEventElapsedTime(&gpuTime, startEvent, endEvent);
    

	/*printf("Matrix addition: \n");
	for (int r = 0; r < N; r++){
		for (int c = 0; c < N; c++)
			printf("%f ", result[r*N + c]);

		printf("\n");
	}*/
    
    //CPU equivalent
    auto cpustart = std::chrono::system_clock::now();
    cpu_add_matrix(ma, mb, cpuresult);
    auto cpuend = std::chrono::system_clock::now();
    std::chrono::duration<double> cpuTime = cpuend-cpustart;

    printf("\nMatrix: %ix%i \nBlocksize: %ix%i \n", N, N, blocksize1d, blocksize1d);
    printf("GPU compute time: %f \n", gpuTime);
    printf("CPU compute time: %f \n", cpuTime.count());

    cudaFree( cd );
    cudaFree( gpu_ma );
	cudaFree( gpu_mb );
	cudaFree( gpu_result );
	printf("done\n");
	return EXIT_SUCCESS;
}
