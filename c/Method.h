#ifndef METHOD
#define METHOD

#include "Image.h"
#include <fftw3.h>
#include <math.h>

#define BLOCK 16

#define PI 3.141592665

#define Complex fftwf_complex

class Method {

        public:
		void FFT();
		//Complex* reconstruct(float DISTANCE) = 0;
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

		fftwf_plan plan;
		fftwf_plan plan1;

		int width;
		int height;

		Method(Image &image) {

			//image.zeroPad(BLOCK);

			width = image.width;
			height = image.height;

        		in = &image;

        		out = new Image(width, height);

			hol_fft = new Complex[width*height];
			work = new Complex[width*height];

			plan = fftwf_plan_dft_2d(height, width,
                                hol_fft, hol_fft,
                                FFTW_FORWARD, FFTW_MEASURE);
			plan1 = fftwf_plan_dft_2d(height, width,
                                work, work,
                                FFTW_BACKWARD, FFTW_MEASURE);

			FFT();
		}
};

void Method::FFT() {

	for (int i=0; i<height; i++)
		for (int j=0; j<width; j++) {
			hol_fft[i*width+j][0] = in->pixel[i*width+j];
			hol_fft[i*width+j][1] = 0;
		} 	


    	fftwf_execute(plan);

}

void Multiply_By_Quadratic_Phase_Factor(Complex *src, Complex *dst, float constPhase, Complex fieldStep, int width, int height);
void Multiply_By_Quadratic_Phase_Factor_shift(Complex *src, Complex *dst, float constPhase, Complex fieldStep, int width, int height);

void Method::reconstruct_from_FFT(float dist) {

        float constPhase = (-PI*WAVELENGTH*dist);

	Complex fieldStep;
	fieldStep[0]=1.f/(width*PIXEL_PITCHx);
	fieldStep[1]=1.f/(height*PIXEL_PITCHy);

	Multiply_By_Quadratic_Phase_Factor_shift(hol_fft, work, constPhase, fieldStep, width, height);

    	fftwf_execute(plan1);
}

void Method::getResult() {

	for (int i=0; i<height; i++)
                for (int j=0; j<width; j++) {
                        float x=work[i*width+j][0];
                        float y=work[i*width+j][1];
                        out->pixel[i*width+j] = sqrt(x*x + y*y);
                }

}

void Multiply_By_Quadratic_Phase_Factor(Complex *src, Complex *dst, float constPhase, Complex fieldStep, int width, int height) {

  float scale = 0.7f/float(width*height);

  for (int y=0; y<height; y++) {
	for (int x=0; x<width; x++) {

  		int a = y*width+x;

  		Complex Position;
  		Complex fieldUpperLeft;
  		fieldUpperLeft[0] = width/2;
  		fieldUpperLeft[1] = height/2;
  		Position[0] = (-1*fieldUpperLeft[0] + x)*fieldStep[0];
  		Position[1] = (-1*fieldUpperLeft[1] + y)*fieldStep[1];

  		float p = (Position[0]*Position[0] + Position[1]*Position[1])*constPhase;

  		Complex e;
		e[0] = cos(p);
		e[1] = sin(p);

  		dst[a][0] = (src[a][0] * e[0] - src[a][1] * e[1])*scale; 
		dst[a][1] = (src[a][0] * e[1] + src[a][1] * e[0])*scale;

	}
  }

}


void Multiply_By_Quadratic_Phase_Factor_shift(Complex *src, Complex *dst, float constPhase, Complex fieldStep, int width, int height) {

	float scale = 0.7f/float(width*height);

	int halfWidth = width/2;
	int halfHeight = height/2;

	for (int y1=0; y1<halfHeight; y1++) {
        for (int x1=0; x1<halfWidth; x1++) {

  	int x2 = x1 + halfWidth;
  	int y2 = y1;

  	int x3 = x1 + halfWidth;
  	int y3 = y1 + halfHeight;

  	int x4 = x1;
  	int y4 = y1 + halfHeight;

  	int nQuadrant_1 = y1*width+x1;
  	int nQuadrant_2 = y2*width+x2;
  	int nQuadrant_3 = y3*width+x3;
  	int nQuadrant_4 = y4*width+x4;

  	Complex Position;
  	Complex fieldUpperLeft;
  	fieldUpperLeft[0] = halfWidth;
  	fieldUpperLeft[1] = halfHeight;

	Position[0] = (-1.f*fieldUpperLeft[0] + x1)*fieldStep[0];
  	Position[1] = (-1.f*fieldUpperLeft[1] + y1)*fieldStep[1];

  	float p = (Position[0]*Position[0] + Position[1]*Position[1])*constPhase;

	Complex e;
        e[0] = cos(p)*scale;
        e[1] = sin(p)*scale;

  	dst[nQuadrant_3][0] = src[nQuadrant_3][0] * e[0] - src[nQuadrant_3][1] * e[1];
  	dst[nQuadrant_3][1] = src[nQuadrant_3][0] * e[1] - src[nQuadrant_3][1] * e[0];

  	Position[0] = (-1.f*fieldUpperLeft[0] + x3)*fieldStep[0];
  	Position[1] = (-1.f*fieldUpperLeft[1] + y3)*fieldStep[1];

  	p = (Position[0]*Position[0] + Position[1]*Position[1])*constPhase;

        e[0] = cos(p)*scale;
        e[1] = sin(p)*scale;

  	dst[nQuadrant_1][0] = src[nQuadrant_1][0] * e[0] - src[nQuadrant_1][1] * e[1];
  	dst[nQuadrant_1][1] = src[nQuadrant_1][0] * e[1] - src[nQuadrant_1][1] * e[0];

  	Position[0] = (-1.f*fieldUpperLeft[0] + x2)*fieldStep[0];
  	Position[1] = (-1.f*fieldUpperLeft[1] + y2)*fieldStep[1];

  	p = (Position[0]*Position[0] + Position[1]*Position[1])*constPhase;

        e[0] = cos(p)*scale;
        e[1] = sin(p)*scale;

  	dst[nQuadrant_4][0] = src[nQuadrant_4][0] * e[0] - src[nQuadrant_4][1] * e[1];
  	dst[nQuadrant_4][1] = src[nQuadrant_4][0] * e[1] - src[nQuadrant_4][1] * e[0];

  	Position[0] = (-1.f*fieldUpperLeft[0] + x4)*fieldStep[0];
  	Position[1] = (-1.f*fieldUpperLeft[1] + y4)*fieldStep[1];

  	p = (Position[0]*Position[0] + Position[1]*Position[1])*constPhase;

        e[0] = cos(p)*scale;
        e[1] = sin(p)*scale;

  	dst[nQuadrant_2][0] = src[nQuadrant_2][0] * e[0] - src[nQuadrant_2][1] * e[1];
  	dst[nQuadrant_2][1] = src[nQuadrant_2][0] * e[1] - src[nQuadrant_2][1] * e[0];

	}
	}
}

#endif

