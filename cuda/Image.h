
#include "tiffio.h"
//#include "tchar.h"

#include <iostream>

using namespace std;


class Image {
	public:

		int width;
		int height;

		int padX;
		int padY;

		unsigned char *pixel;
		unsigned char *d_pixel;

		char *outFile;

		Image() {
			width=0;
			height=0;
			outFile = NULL;
		}

		Image(int width, int height) {
			this->width = width;
			this->height = height;
			pixel = (unsigned char *) malloc(sizeof(unsigned char)*width*height);
                        cudaMalloc( (void**) &d_pixel, sizeof(unsigned char) * width*height);
			outFile = NULL;
		}

		~Image() {
			free(pixel);
		}

		void zeroPad(int BLOCK);
		void readTiff(const char *filename);
		void writeTiff();
		void writeTiff(const char *filename);
		void setZero();
                void copyTo_device();
		void copyTo_host();
};

void Image::setZero() {

	for (int i=0; i<width*height; i++)
		pixel[i]=0;
}

void Image::zeroPad(int BLOCK) {

	int padx = width%(BLOCK);
        if (padx >0) {
                padx = BLOCK-padx;
		width += padx;
	}
	int pady = height%(BLOCK);
        if (pady >0) {
		pady = BLOCK-pady;
                height += pady;
	}
	if (padx>0 || pady >0) {
		unsigned char *newpixel = (unsigned char *) malloc(sizeof(unsigned char)*width*height);
		if (padx>0) {
			int oldwidth=width-padx;
			for (int i=0; i<height-pady; i++) {
				for (int j=0; j<oldwidth; j++)
					newpixel[i*width+j]=pixel[i*oldwidth+j];
				for (int j=0; j<padx; j++)
                                        newpixel[i*width+j+oldwidth]=0;
			}	
		} else {
			for (int i=0; i<width*(height-pady); i++)
				newpixel[i] = pixel[i];
		}
		if (pady>0)
			for (int i=height-pady; i<height; i++)
				for (int j=0; j<width; j++)
					newpixel[i*width+j]=0;

		pixel = newpixel;
	}	
	

}

int setZeroPAD(int length, int BLOCK) {

        int pad = length%(BLOCK);
        if (pad >0)
                pad = BLOCK-pad;

	return length+pad;
}

void Image::readTiff(const char* filename) {

	
	TIFF* tif = TIFFOpen(filename, "r");
	
	TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);           // int width;
	TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);        // int height;

	//width = 2592; //1024 //2040 //2592
	//height = 1944; //1024 //1088 //1944

	int npixels=width*height;
	unsigned char *image;
	image = (unsigned char *) _TIFFmalloc(npixels *sizeof(unsigned char));

	for (int row = 0; row < height; row++)
	{
		if (TIFFReadScanline(tif, image + row*width, row, 0) < 0)
		break;
	}
    	TIFFClose(tif);

		this->pixel = image;
}

void Image::writeTiff() {

	if (outFile != NULL)
		writeTiff(outFile);
}

void Image::writeTiff(const char *filename) {

	int npixels=width*height;

	//WRITE IMAGE
	TIFF* out = TIFFOpen(filename, "w");

	int sampleperpixel = 1;

	TIFFSetField(out, TIFFTAG_IMAGEWIDTH, width);  // set the width of the image
	TIFFSetField(out, TIFFTAG_IMAGELENGTH, height);    // set the height of the image
	TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, sampleperpixel);   // set number of channels per pixel
	TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, 8);    // set the size of the channels
	TIFFSetField(out, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);    // set the origin of the image.
	//Some other essential fields to set that you do not have to understand for now.
	TIFFSetField(out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
	TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK); 

	tsize_t linebytes = sampleperpixel * width;     // length in memory of one row of pixels in the image. 
	//We set the strip size of the file to be size of one row of pixels
	TIFFSetField(out, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(out, width*sampleperpixel));

	//Now writing image to the file one strip at a time
	for (int row = 0; row < height; row++)
	{
		if (TIFFWriteScanline(out, pixel + row*width, row, 0) < 0)
		break;
	}
	
	(void) TIFFClose(out);
        
}

void Image::copyTo_device() {

	if (d_pixel==NULL) 
		cudaMalloc( (void**) &d_pixel, sizeof(unsigned char) * width*height);

	cudaMemcpy( d_pixel, pixel, sizeof(unsigned char) * width*height,
                                cudaMemcpyHostToDevice);
}

void Image::copyTo_host() {

	cudaMemcpy( pixel, d_pixel, sizeof(unsigned char) * width*height,
                                cudaMemcpyDeviceToHost);
}
