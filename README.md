## Holographic Image Reconstruction Library

Holographic image reconstruction at specific distance.

Generation of z-map and infocus image using a stack of reconstructed
images at various distances.

Using JNI to access fftw (or CuFFT) and tiff libraries (only works with tiff images).

See .classpath file to see necessary external java libraries.
(ImageJ for plotting images and JZY3D for 3D plots)

First compile the native c codes (or cuda codes for GPU acceleration) and copy resulting libholography.so file into java classpath.
(Provided libholography.so is c version for Linux amd64 platform.)

If using eclipse define environment variable LD_PRELOAD=/path/to/libtiff.so:/path/to/libfftw3f.so.3

Sample holographic image:

![Alt text](https://github.com/dugannaz/Holography/blob/master/image.png "Holographic image")

Reconstructed image:

![Alt text](https://github.com/dugannaz/Holography/blob/master/reconstructed.png "Reconstructed image")
