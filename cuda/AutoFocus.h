#ifndef AUTOFOCUS
#define AUTOFOCUS

#include "Method.h"

class AutoFocus : public Method {

        public:

		void mergeFocus(float startDist, float step, float n);
		void createFilter(int filterSize);
		void varianceConv();
		//void map(float *var, float *maxVar, int z);

		float *filter;
		Complex *filter_F;
		Complex *work1;
		float *variance_in;
		float *mean;
		float *variance;
		float *d_variance;
		float *d_variance_in;
		float *d_mean;
		Complex *d_filter_F;
		Complex *d_work1;
		
		int SIGNAL_SIZE;
		int HALF_SIZE;

		cufftHandle planR2C;
		cufftHandle planC2R;

		AutoFocus(Image &image, int filterSize) 
		:Method(image)
		{
			
			SIGNAL_SIZE = width*height;
			HALF_SIZE = (width/2+1)*height;
			variance_in = new float[width*height];
			variance = new float[width*height];
			mean = new float[width*height];
			filter = new float[width*height];
			filter_F = new Complex[(width/2+1)*height];
			work1 = new Complex[(width/2+1)*height];

			cudaMalloc( (void**) &d_variance_in, sizeof(float) * SIGNAL_SIZE);
                        cudaMalloc( (void**) &d_mean, sizeof(float) * SIGNAL_SIZE);
                        cudaMalloc( (void**) &d_variance, sizeof(float) * SIGNAL_SIZE);
                        cudaMalloc( (void**) &d_filter_F, sizeof(Complex) * HALF_SIZE);
                        cudaMalloc( (void**) &d_work1, sizeof(Complex) * HALF_SIZE);


			cufftPlan2d(&planR2C, height, width, CUFFT_R2C);
                        cufftPlan2d(&planC2R, height, width, CUFFT_C2R);
 
			createFilter(filterSize);
			//out->setZero();	
		}
};

__global__ void map(float *var, float *maxVar, unsigned char *out, int z);

void AutoFocus::mergeFocus(float startDist, float step, float n) {

	float *maxVar;
	cudaMalloc( (void**) &maxVar, sizeof(float) * SIGNAL_SIZE);
	cudaMemset(&maxVar, 0, sizeof(float) * SIGNAL_SIZE);

	float DISTANCE=startDist;

        for (int i=0; i<n; i++) {

		DISTANCE += step;

                reconstruct_from_FFT(DISTANCE);
                varianceConv();
		map<<<grid, block>>>(d_variance, maxVar, out->d_pixel, i);
		out->copyTo_host();
        }

}


__global__ void map(float *var, float *maxVar, unsigned char *out, int z)
{
  int2 PositionI;

  PositionI.x = blockIdx.x*blockDim.x + threadIdx.x;
  PositionI.y = blockIdx.y*blockDim.y + threadIdx.y;
  int a = PositionI.y*blockDim.x*gridDim.x+PositionI.x;

  if (var[a]>maxVar[a]) {
          out[a] = z;
          maxVar[a] = var[a];
  }
}


__global__ void setSquare(float *src, int N)
{

  int2 PositionI;

  PositionI.x = blockIdx.x*blockDim.x + threadIdx.x;
  PositionI.y = blockIdx.y*blockDim.y + threadIdx.y;
  int a = PositionI.y*blockDim.x*gridDim.x+PositionI.x;

  if (PositionI.x < N && PositionI.y < N)
        src[a] = 1.f/float(N*N);
  else
    src[a] = 0.f;
}


void AutoFocus::createFilter(int filterSize) {


	float* d_filter;
        cudaMalloc( (void**) &d_filter, sizeof(float) * SIGNAL_SIZE);

	setSquare<<<grid,block>>>(d_filter,filterSize);
        cufftExecR2C(planR2C, (cufftReal *)d_filter, (cufftComplex *)d_filter_F);

        cudaFree(d_filter);

}

__global__ void Multiply_2_Arrays_MultScal(Complex * A, Complex * B, int width, float scal)
{
  int2 PositionI;

  PositionI.x = blockIdx.x*blockDim.x + threadIdx.x;
  PositionI.y = blockIdx.y*blockDim.y + threadIdx.y;
  int a = PositionI.y*width+PositionI.x;

  if (PositionI.x < width)
  A[a] = make_float2((A[a].x * B[a].x - A[a].y * B[a].y)*scal, (A[a].x*B[a].y + A[a].y*B[a].x)*scal);
}

__global__ void subtractMeanABS(float *data, float *mean)
{
  int2 PositionI;

  PositionI.x = blockIdx.x*blockDim.x + threadIdx.x;
  PositionI.y = blockIdx.y*blockDim.y + threadIdx.y;
  int a = PositionI.y*blockDim.x*gridDim.x+PositionI.x;

  float subtract = data[a]-mean[a];

  data[a] = abs(subtract);

}

__global__ void DivideCopyTo_new_Array(float *src, float *src2, float *dst)
{

  int2 PositionI;

  PositionI.x = blockIdx.x*blockDim.x + threadIdx.x;
  PositionI.y = blockIdx.y*blockDim.y + threadIdx.y;
  int a = PositionI.y*blockDim.x*gridDim.x+PositionI.x;

  dst[a] = src[a]/(src2[a]*src2[a]);
}


/* Abosolute variance autofocus metric calculation
 */
void AutoFocus::varianceConv() {

	CopyTo_New_Array<<<grid, block>>>(work, d_variance_in);
	
	// mean calculation
	cufftExecR2C(planR2C, (cufftReal *)d_variance_in, (cufftComplex *)d_work1);

	int Nx = width/2+1;
	grid.x = Nx / block.x +1;
        Multiply_2_Arrays_MultScal<<<grid, block>>>(d_work1, d_filter_F, Nx, 1.0f/(float(SIGNAL_SIZE)));
        grid.x = width / block.x;
	
	cufftExecC2R(planC2R, (cufftComplex *)d_work1, (cufftReal *)d_mean);

	// abs(I-m)
	subtractMeanABS<<<grid, block>>>(d_variance_in, d_mean);

        // variance calculation
	cufftExecR2C(planR2C, (cufftReal *)d_variance_in, (cufftComplex *)d_work1);

	grid.x = Nx / block.x +1;
        Multiply_2_Arrays_MultScal<<<grid, block>>>(d_work1, d_filter_F, Nx, 2000.f/(float(SIGNAL_SIZE)));
        grid.x = width / block.x;

	cufftExecC2R(planC2R, (cufftComplex *)d_work1, (cufftReal *)d_variance);

	// copy back
	DivideCopyTo_new_Array<<<grid, block>>>(d_variance, d_mean, d_variance);

}

#endif

