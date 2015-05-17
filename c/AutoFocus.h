#ifndef AUTOFOCUS
#define AUTOFOCUS

#include "Method.h"

class AutoFocus : public Method {

        public:

		void mergeFocus(float startDist, float step, float n);
		void createFilter(int filterSize);
		void varianceConv();
		void map(float *var, float *maxVar, int z);

		float *filter;
		Complex *filter_F;
		Complex *work1;
		float *variance_in;
		float *mean;
		float *variance;
		

		fftwf_plan plan2;
		fftwf_plan plan3;
		fftwf_plan plan4;
		fftwf_plan plan5;

		AutoFocus(Image &image, int filterSize) 
		:Method(image)
		{
			variance_in = new float[width*height];
			variance = new float[width*height];
			mean = new float[width*height];
			filter = new float[width*height];
			filter_F = new Complex[(width/2+1)*height];
			work1 = new Complex[(width/2+1)*height];

			plan2 = fftwf_plan_dft_r2c_2d(height, width,
                                variance_in, work1,
                                FFTW_MEASURE);
			plan3 = fftwf_plan_dft_c2r_2d(height, width,
                                work1, mean,
                                FFTW_MEASURE);
			plan5 = fftwf_plan_dft_c2r_2d(height, width,
                                work1,variance,
                                FFTW_MEASURE);
			plan4 = fftwf_plan_dft_r2c_2d(height, width,
                                filter, filter_F,
                                FFTW_MEASURE);
			
			createFilter(filterSize);
			//out->setZero();	
		}
};

void AutoFocus::mergeFocus(float startDist, float step, float n) {

	float *maxVar = new float[width*height];
	for (int i=0; i<width*height; i++)
		maxVar[i]=0.f;

	float DISTANCE=startDist;

        for (int i=0; i<n; i++) {

		DISTANCE += step;

                reconstruct_from_FFT(DISTANCE);
                varianceConv();
		map(variance,maxVar,i);
        }

}

void AutoFocus::map(float *var, float *maxVar, int z) {

	for (int i=0; i<height; i++)
                for (int j=0; j<width; j++) {
			int xy=i*width+j;
			if (var[xy]>maxVar[xy]) {
				out->pixel[xy]=z;
				maxVar[xy]=var[xy];
			}
		}
}

void AutoFocus::createFilter(int filterSize) {

	float val=1.f/float(filterSize*filterSize);

	for (int i=0; i<height; i++)
		for (int j=0; j<width; j++) {
			if (i<filterSize && j<filterSize)
				filter[i*width+j]=val;
			else
				filter[i*width+j]=0.f; 
	}
	fftwf_execute(plan4);
}

void AutoFocus::varianceConv() {

	for (int i=0; i<height; i++)
                for (int j=0; j<width; j++) {
                        float x=work[i*width+j][0];
                        float y=work[i*width+j][1];
                        variance_in[i*width+j] = sqrt(x*x + y*y);
                }



        // mean calculation
        //cufftExecR2C(planR2C, (cufftReal *)d_input, (cufftComplex *)d_input_F);
	fftwf_execute(plan2);

        int Nx = width/2+1;
	float scale=1.f/float(width*height);

	for (int i=0; i<height; i++)
                for (int j=0; j<Nx; j++) {
			work1[i*Nx+j][0]*=(filter_F[i*Nx+j][0]*scale);
			work1[i*Nx+j][1]*=(filter_F[i*Nx+j][1]*scale);
	}

        //grid.x = (in_img.width/2+1) / block.x +1;
        //Multiply_2_Arrays_MultScal<<<grid,block,0,reconst_method.recStream>>>(d_input_F, d_filter_F, Nx, 1.0f/(float(SIGNAL_SIZE)));
        //grid.x = in_img.width / block.x;

        //cufftExecC2R(planC2R, (cufftComplex *)d_input_F, (cufftReal *)d_mean);
	fftwf_execute(plan3);

        // abs(I-m)
        //subtractMeanABS<<<grid,block,0,reconst_method.recStream>>>(d_input, d_mean);
	for (int i=0; i<height; i++)
                for (int j=0; j<width; j++) 
			variance_in[i*width+j]=fabs(variance_in[i*width+j]-mean[i*width+j]);

        // variance calculation

        //cufftExecR2C(planR2C, (cufftReal *)d_input, (cufftComplex *)d_input_F);
        fftwf_execute(plan2);

        //grid.x = (in_img.width/2+1) / block.x +1;

        //Multiply_2_Arrays_MultScal<<<grid,block,0,reconst_method.recStream>>>(d_input_F, d_filter_F, Nx, 2000.f/(float(SIGNAL_SIZE)));
        scale=2000.f/float(width*height);

        for (int i=0; i<height; i++)
                for (int j=0; j<Nx; j++) {
                        work1[i*Nx+j][0]*=(filter_F[i*Nx+j][0]*scale);
                        work1[i*Nx+j][1]*=(filter_F[i*Nx+j][1]*scale);
        }


	fftwf_execute(plan5);
        //cufftExecC2R(planC2R, (cufftComplex *)d_input_F, (cufftReal *)d_var);

        //grid.x = reconst_method.in_img.width / block.x;

        // copy back
        //DivideCopyTo_new_Array<<<grid, block,0,reconst_method.recStream>>>(d_var, d_mean, out_img->d_pixel);
	for (int i=0; i<height; i++)
                for (int j=0; j<width; j++)
			variance[i*width+j]/=(mean[i*width+j]*mean[i*width+j]);
}

#endif

