#ifndef METHOD
#define METHOD

#include "Image.h"
#include <math.h>
#include "cufft.h"
#include "cuda.h"
#include "cuda_runtime_api.h"

#define BLOCK 4

#define PI 3.141592665

#define Complex cufftComplex

class Method {

        public:
		void FFT();
                void reconstruct_from_FFT(float dist);
		void getResult();

		float WAVELENGTH;

		float PIXEL_PITCHx;
                float PIXEL_PITCHy;

		Image *in;
		Image *out;
		Complex *hol_fft;
		Complex *work;
		float *variance_in;

		cufftHandle plan;

		int width;
		int height;

		dim3 block;
		dim3 grid;
		dim3 halfGrid;

		Method(Image &image) {

			//image.zeroPad(BLOCK);

        		in = &image;
			width = image.width;
			height = image.height;

			block.x = BLOCK;
			block.y = BLOCK;

			grid.x = in->width / block.x;
			grid.y = in->height / block.y;
			halfGrid.x = in->width / (2*block.x);
			halfGrid.y = in->height / (2*block.y);


        		out = new Image(width, height);
			out->setZero();
        		out->copyTo_device();

                        cudaMalloc( (void**) &hol_fft, sizeof(Complex) * width*height);
                        cudaMalloc( (void**) &work, sizeof(Complex) * width*height);

			int n1d[3]= {width, height, 1};
		
			cufftPlan2d(&plan, height, width, CUFFT_C2C);

			FFT();
		}
};

__global__ void CopyTo_New_Array(unsigned char *src, Complex *dst);
__global__ void CopyTo_New_Array(Complex *src, unsigned char *dst);
__global__ void CopyTo_New_Array(unsigned char *src, unsigned char *dst);
__global__ void Multiply_By_Quadratic_Phase_Factor_shift(Complex *src, Complex *dst, float constPhase, Complex fieldStep, int signalSize);


// initial FFT of spectral reconstruction  method
void Method::FFT() {

	in->copyTo_device();
        CopyTo_New_Array<<<grid, block>>>(in->d_pixel, hol_fft);

	cufftExecC2C(plan, hol_fft, hol_fft, CUFFT_FORWARD);
}

// rest of spectral reconstruction method
void Method::reconstruct_from_FFT(float dist) {

        float constPhase = (-PI*WAVELENGTH*dist);

	Complex fieldStep;
	fieldStep.x=1.f/(width*PIXEL_PITCHx);
	fieldStep.y=1.f/(height*PIXEL_PITCHy);

        Multiply_By_Quadratic_Phase_Factor_shift<<<halfGrid,block>>>(hol_fft, work, constPhase, fieldStep, width*height);

	cufftExecC2C(plan, work, work, CUFFT_INVERSE);
}

// copy final result into output image array
void Method::getResult() {

	CopyTo_New_Array<<<grid, block>>>(work, out->d_pixel);

	out->copyTo_host();
}

__global__ void CopyTo_New_Array(unsigned char *src, Complex *dst)
{
  int2 PositionI;

  PositionI.x = blockIdx.x*blockDim.x + threadIdx.x;
  PositionI.y = blockIdx.y*blockDim.y + threadIdx.y;

  int a = PositionI.y*blockDim.x*gridDim.x+PositionI.x;
  dst[a] = make_float2(src[a],0);
}

__global__ void CopyTo_New_Array(Complex *src, unsigned char *dst)
{
  int2 PositionI;

  PositionI.x = blockIdx.x*blockDim.x + threadIdx.x;
  PositionI.y = blockIdx.y*blockDim.y + threadIdx.y;

  int a = PositionI.y*blockDim.x*gridDim.x+PositionI.x;
  float x = src[a].x;
  float y = src[a].y;
  dst[a] = sqrt(x*x + y*y);
}

__global__ void CopyTo_New_Array(Complex *src, float *dst)
{
  int2 PositionI;

  PositionI.x = blockIdx.x*blockDim.x + threadIdx.x;
  PositionI.y = blockIdx.y*blockDim.y + threadIdx.y;

  int a = PositionI.y*blockDim.x*gridDim.x+PositionI.x;
  float x = src[a].x;
  float y = src[a].y;
  dst[a] = sqrt(x*x + y*y);
}

__global__ void CopyTo_New_Array(unsigned char *src, unsigned char *dst)
{
  int2 PositionI;

  PositionI.x = blockIdx.x*blockDim.x + threadIdx.x;
  PositionI.y = blockIdx.y*blockDim.y + threadIdx.y;

  int a = PositionI.y*blockDim.x*gridDim.x+PositionI.x;
  dst[a] = src[a];
}


__global__ void Multiply_By_Quadratic_Phase_Factor_shift(Complex * src, Complex * dst, float constPhase, Complex fieldStep, int signalSize)
{

  float scale = 0.7f/float(signalSize);

  int2 PositionQuadrant_1;
  PositionQuadrant_1.x = blockIdx.x*blockDim.x + threadIdx.x;
  PositionQuadrant_1.y = blockIdx.y*blockDim.y + threadIdx.y;

  int2 PositionQuadrant_2;
  PositionQuadrant_2.x = PositionQuadrant_1.x + gridDim.x*blockDim.x;
  PositionQuadrant_2.y = PositionQuadrant_1.y;

  int2 PositionQuadrant_3;
  PositionQuadrant_3.x = PositionQuadrant_1.x + gridDim.x*blockDim.x;
  PositionQuadrant_3.y = PositionQuadrant_1.y + gridDim.y*blockDim.y;

  int2 PositionQuadrant_4;
  PositionQuadrant_4.x = PositionQuadrant_1.x;
  PositionQuadrant_4.y = PositionQuadrant_1.y + gridDim.y*blockDim.y;

  int nQuadrant_1 = PositionQuadrant_1.y*blockDim.x*gridDim.x*2+PositionQuadrant_1.x;
  int nQuadrant_2 = PositionQuadrant_2.y*blockDim.x*gridDim.x*2+PositionQuadrant_2.x;
  int nQuadrant_3 = PositionQuadrant_3.y*blockDim.x*gridDim.x*2+PositionQuadrant_3.x;
  int nQuadrant_4 = PositionQuadrant_4.y*blockDim.x*gridDim.x*2+PositionQuadrant_4.x;

  Complex Position;
  Complex fieldUpperLeft;
  fieldUpperLeft.x = blockDim.x*gridDim.x;
  fieldUpperLeft.y = blockDim.y*gridDim.y;

  Position.x = (-1*fieldUpperLeft.x + PositionQuadrant_1.x)*fieldStep.x;
  Position.y = (-1*fieldUpperLeft.y + PositionQuadrant_1.y)*fieldStep.y;

  float p = (Position.x*Position.x + Position.y*Position.y)*constPhase;

  Complex e = make_float2(cos(p)*scale,sin(p)*scale);

  dst[nQuadrant_3] = make_float2(src[nQuadrant_3].x * e.x - src[nQuadrant_3].y * e.y, 
	  src[nQuadrant_3].x*e.y + src[nQuadrant_3].y*e.x);

  Position.x = (-1*fieldUpperLeft.x + PositionQuadrant_3.x)*fieldStep.x;
  Position.y = (-1*fieldUpperLeft.y + PositionQuadrant_3.y)*fieldStep.y;

  p = (Position.x*Position.x + Position.y*Position.y)*constPhase;

  e = make_float2(cos(p)*scale,sin(p)*scale);

  dst[nQuadrant_1] = make_float2(src[nQuadrant_1].x * e.x - src[nQuadrant_1].y * e.y, 
	  src[nQuadrant_1].x*e.y + src[nQuadrant_1].y*e.x);

  Position.x = (-1*fieldUpperLeft.x + PositionQuadrant_2.x)*fieldStep.x;
  Position.y = (-1*fieldUpperLeft.y + PositionQuadrant_2.y)*fieldStep.y;

  p = (Position.x*Position.x + Position.y*Position.y)*constPhase;

  e = make_float2(cos(p)*scale,sin(p)*scale);

  dst[nQuadrant_4] = make_float2(src[nQuadrant_4].x * e.x - src[nQuadrant_4].y * e.y, 
	  src[nQuadrant_4].x*e.y + src[nQuadrant_4].y*e.x);

  Position.x = (-1*fieldUpperLeft.x + PositionQuadrant_4.x)*fieldStep.x;
  Position.y = (-1*fieldUpperLeft.y + PositionQuadrant_4.y)*fieldStep.y;

  p = (Position.x*Position.x + Position.y*Position.y)*constPhase;

  e = make_float2(cos(p)*scale,sin(p)*scale);

  dst[nQuadrant_2] = make_float2(src[nQuadrant_2].x * e.x - src[nQuadrant_2].y * e.y, 
	  src[nQuadrant_2].x*e.y + src[nQuadrant_2].y*e.x);
}



#endif

