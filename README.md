## Holographic Image Reconstruction Library

Holographic image reconstruction at specific distance.

Generation of z-map and infocus image using a stack of reconstructed
images at various distances.

Using JNI to access fftw and tiff libraries (only works with tiff images).

See .classpath file to see necessary external java libraries.
(ImageJ for plotting images and JZY3D for 3D plots)

First compile the native c codes and copy resulting libholography.so
file into java classpath.

Sample holographic image:

![Alt text](https://github.com/dugannaz/Holography/blob/master/image.png "Holographic image")

Reconstructed image:

![Alt text](https://github.com/dugannaz/Holography/blob/master/reconstructed.png "Reconstructed image")
