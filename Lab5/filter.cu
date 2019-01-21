// Lab 5, image filters with CUDA.

// Compile with a command-line similar to Lab 4:
// nvcc filter.cu -c -arch=sm_30 -o filter.o
// g++ filter.o milli.c readppm.c -lGL -lm -lcuda -lcudart -L/usr/local/cuda/lib -lglut -o filter
// or (multicore lab)
// nvcc filter.cu -c -arch=sm_20 -o filter.o
// g++ filter.o milli.c readppm.c -lGL -lm -lcuda -L/usr/local/cuda/lib64 -lcudart -lglut -o filter

// 2017-11-27: Early pre-release, dubbed "beta".
// 2017-12-03: First official version! Brand new lab 5 based on the old lab 6.
// Better variable names, better prepared for some lab tasks. More changes may come
// but I call this version 1.0b2.
// 2017-12-04: Two fixes: Added command-lines (above), fixed a bug in computeImages
// that allocated too much memory. b3
// 2017-12-04: More fixes: Tightened up the kernel with edge clamping.
// Less code, nicer result (no borders). Cleaned up some messed up X and Y. b4

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#ifdef __APPLE__
  #include <GLUT/glut.h>
  #include <OpenGL/gl.h>
#else
  #include <GL/glut.h>
#endif
#include "readppm.h"
#include "milli.h"

// Use these for setting shared memory size.
#define maxKernelSizeX 32
#define maxKernelSizeY 32

//#define median
 #define separable
//#define gaussian

__device__ int find_median(int *data, int numelements)
{
	int sum = 0;
	int iter = 0;
	int midvalue = (int)((numelements+1)/2);	// numelements will always be odd

	while(sum < midvalue)
	{
		sum += data[iter++];
	}
	return iter;
}

__global__ void filter(unsigned char *image, unsigned char *out, const unsigned int imagesizex, const unsigned int imagesizey, const int kernelsizex, const int kernelsizey)
{ 

	//
	// Load image to shared memory
	//

	// map from blockIdx to pixel position
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// allocate memory requred for maximum kernel size
	const int maxmemsizex = 2 * maxKernelSizeX + 1;
	const int maxmemsizey = 2 * maxKernelSizeY + 1;
	__shared__ unsigned char thisBlock[maxmemsizex * maxmemsizey * 3];

	// define the block that should be loaded to shared memory
	// TODO Shoud memstartx and medendx be multiplied with 3?
	int memstartx = max(0, (int)(blockIdx.x*blockDim.x) - kernelsizex);
	int memstarty = max(0, (int)(blockIdx.y*blockDim.y) - kernelsizey);
	int memendx = min(imagesizex-1, memstartx + (int)blockDim.x + 2*kernelsizex - 1);
	int memendy = min(imagesizey-1, memstarty + (int)blockDim.y + 2*kernelsizey - 1);

	// how much memory should each tread load
	int memloadsize = (memendx - memstartx + 1) * (memendy - memstarty + 1);
	int blocksize = blockDim.x * blockDim.y;
	int memperthread = (int)(memloadsize/(blocksize));

	int memsizex = memendx - memstartx + 1;

	// load image pixels to shared memory
	for(int i = 0; i <= memperthread; i++)
	{
		// Memory image coordinates (in pixels, without rgb)
		int mem_idx = (threadIdx.x + threadIdx.y *memsizex  + i*blocksize);
		int mem_x = mem_idx % memsizex;
		int mem_y = (int)((mem_idx - mem_x) / memsizex);
		// Change mem_idx to work with rgb
		mem_idx *= 3;

		// Corresponding index in image data
		int img_x = memstartx + mem_x;
		int img_y = memstarty + mem_y;
		int img_idx = 3 * (img_x + img_y * imagesizex);
		
		if(mem_idx <= 3*memloadsize)
		{
			// r, g, b
			thisBlock[mem_idx] = image[img_idx];
			thisBlock[mem_idx+1] = image[img_idx+1];
			thisBlock[mem_idx+2] = image[img_idx+2];
		}
	}

	__syncthreads();

	//
	// Apply filter to image (curently not with shared memory)
	//

	int dy, dx;
	unsigned int sumx, sumy, sumz;

	int divby = (2*kernelsizex+1)*(2*kernelsizey+1); // Works for box filters only!

	// x and y in shared memory coordinates
	int memx = x - memstartx;
	int memy = y - memstarty;

	#ifdef gaussian
		int gaussweights[] = {1, 4, 6, 4, 1};
		divby = 16;
	#endif

	if (x < imagesizex && y < imagesizey) // If inside image
	{
		// Filter kernel (simple box filter)

		#ifdef median
			int histogram_x[256];
			int histogram_y[256];
			int histogram_z[256];

			int u;
			for(u = 0; u < 256; u++)
			{
				histogram_x[u] = 0;
				histogram_y[u] = 0;
				histogram_z[u] = 0;
			}
		#endif

		sumx=0;sumy=0;sumz=0;
		for(dy=-kernelsizey;dy<=kernelsizey;dy++)
			for(dx=-kernelsizex;dx<=kernelsizex;dx++)	
			{
				// Use max and min to avoid branching!
				int yy = min(max(memy+dy, 0), memendy);
				int xx = min(max(memx+dx, 0), memendx);

				int idx = 3* (xx + memsizex*yy);
				#ifdef gaussian
					// dx or dy will always be == 0 when using separable filter
					int weight = gaussweights[dx+dy+2];
					sumx += weight * thisBlock[idx];
					sumy += weight * thisBlock[idx+1];
					sumz += weight * thisBlock[idx+2];
				#elif defined(median)
					histogram_x[(int)(thisBlock[idx])] += 1;
					histogram_y[(int)(thisBlock[idx+1])] += 1;
					histogram_z[(int)(thisBlock[idx+2])] += 1;
				#else 
					sumx += thisBlock[idx];
					sumy += thisBlock[idx+1];
					sumz += thisBlock[idx+2];
				#endif
			}

		#ifdef median
			out[(y*imagesizex+x)*3+0] = find_median(histogram_x, divby);
			out[(y*imagesizex+x)*3+1] = find_median(histogram_y, divby);
			out[(y*imagesizex+x)*3+2] = find_median(histogram_z, divby);			
		#else
			out[(y*imagesizex+x)*3+0] = sumx/divby;
			out[(y*imagesizex+x)*3+1] = sumy/divby;
			out[(y*imagesizex+x)*3+2] = sumz/divby;
		#endif
	}
}

// Global variables for image data

unsigned char *image, *pixels, *dev_bitmap, *dev_input, *dev_temp;
unsigned int imagesizey, imagesizex; // Image size

////////////////////////////////////////////////////////////////////////////////
// main computation function
////////////////////////////////////////////////////////////////////////////////
void computeImages(int kernelsizex, int kernelsizey)
{
	if (kernelsizex > maxKernelSizeX || kernelsizey > maxKernelSizeY)
	{
		printf("Kernel size out of bounds!\n");
		return;
	}
  
  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  int blocksize = 4;
  
	pixels = (unsigned char *) malloc(imagesizex*imagesizey*3);
	cudaMalloc( (void**)&dev_input, imagesizex*imagesizey*3);
	cudaMemcpy( dev_input, image, imagesizey*imagesizex*3, cudaMemcpyHostToDevice );
	cudaMalloc( (void**)&dev_bitmap, imagesizex*imagesizey*3);
  
  
  cudaMalloc( (void**)&dev_temp, imagesizex*imagesizey*3);
  cudaEventRecord(start);
  cudaEventSynchronize(start);
	#ifdef separable
    dim3 grid1(imagesizex/(blocksize), imagesizey);
    dim3 grid2(imagesizex*3, imagesizey/blocksize);
    dim3 blockGrid1(blocksize, 1);
    dim3 blockGrid2(3*1, blocksize);
		filter<<<grid1, blockGrid1>>>(dev_input, dev_temp, imagesizex, imagesizey, kernelsizex, 0); // Awful load balance
		filter<<<grid2, blockGrid2>>>(dev_temp, dev_bitmap, imagesizex, imagesizey, 0, kernelsizey); // Awful load balance

	#else
    dim3 grid(imagesizex * 3/(blocksize), imagesizey/blocksize);
    dim3 blockGrid(3*blocksize, blocksize);
		filter<<<grid, blockGrid>>>(dev_input, dev_bitmap, imagesizex, imagesizey, kernelsizex, kernelsizey); // Awful load balance
	#endif
  cudaThreadSynchronize();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
//	Check for errors!
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
	cudaMemcpy( pixels, dev_bitmap, imagesizey*imagesizex*3, cudaMemcpyDeviceToHost );
  float time_taken = 0;

 	 	cudaEventElapsedTime(&time_taken, start, stop);
  	  printf("Time taken time: %f ms\n", time_taken);

	cudaFree( dev_bitmap );
	cudaFree( dev_input );
}

// Display images
void Draw()
{
// Dump the whole picture onto the screen.	
	glClearColor( 0.0, 0.0, 0.0, 1.0 );
	glClear( GL_COLOR_BUFFER_BIT );

	if (imagesizey >= imagesizex)
	{ // Not wide - probably square. Original left, result right.
		glRasterPos2f(-1, -1);
		glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, image );
		glRasterPos2i(0, -1);
		glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE,  pixels);
	}
	else
	{ // Wide image! Original on top, result below.
		glRasterPos2f(-1, -1);
		glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, pixels );
		glRasterPos2i(-1, 0);
		glDrawPixels( imagesizex, imagesizey, GL_RGB, GL_UNSIGNED_BYTE, image );
	}
	glFlush();
}

// Main program, inits
int main( int argc, char** argv) 
{
	glutInit(&argc, argv);
	glutInitDisplayMode( GLUT_SINGLE | GLUT_RGBA );

	if (argc > 1)
		image = readppm(argv[1], (int *)&imagesizex, (int *)&imagesizey);
	else
		image = readppm((char *)"maskros512.ppm", (int *)&imagesizex, (int *)&imagesizey);

	if (imagesizey >= imagesizex)
		glutInitWindowSize( imagesizex*2, imagesizey );
	else
		glutInitWindowSize( imagesizex, imagesizey*2 );
	glutCreateWindow("Lab 5");
	glutDisplayFunc(Draw);

	ResetMilli();

	computeImages(2, 2);

// You can save the result to a file like this:
  writeppm("out.ppm", imagesizey, imagesizex, pixels);
  printf("finish\n");
	glutMainLoop();
	return 0;
}
