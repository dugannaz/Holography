#include<stdio.h>
#include<stdlib.h>
#include<jni.h>
#include "ImageJ.h"
#include "AutoFocus.h"
#include "Holography.h"

jbyteArray as_byte_array(JNIEnv *env, unsigned char* buf, int len) {

    jbyteArray array = env->NewByteArray (len);
    env->SetByteArrayRegion (array, 0, len, reinterpret_cast<jbyte*>(buf));
    return array;
}

unsigned char* as_unsigned_char_array(JNIEnv *env, jbyteArray array) {

    int len = env->GetArrayLength (array);
    unsigned char* buf = new unsigned char[len];
    env->GetByteArrayRegion (array, 0, len, reinterpret_cast<jbyte*>(buf));
    return buf;
}

char* as_char_array(JNIEnv *env, jbyteArray array) {

    int len = env->GetArrayLength (array);
    char* buf = new char[len];
    env->GetByteArrayRegion (array, 0, len, reinterpret_cast<jbyte*>(buf));
    return buf;
}

JNIEXPORT jbyteArray JNICALL Java_Image_readTiff
  (JNIEnv *env, jobject object, jbyteArray file) {

	Image *image = new Image(); 
	image->readTiff((char*)as_unsigned_char_array(env, file));

	return as_byte_array(env, image->pixel, image->width*image->height);

}

JNIEXPORT jbyteArray JNICALL Java_Holography_reconstructAt
(JNIEnv *jnienv, jobject obj, jlong mp, jfloat dist) {

        Method *method = (Method*)mp;

        method->reconstruct_from_FFT(dist);

	method->getResult();

	//if (method->in->width != 2048 || method->in->height != 1088)
	//	method->in = new Image(2048,1088);

	return as_byte_array(jnienv, method->out->pixel, method->in->width*method->in->height);
}

JNIEXPORT jbyteArray JNICALL Java_Holography_zmap
(JNIEnv *jnienv, jobject obj, jlong mp, jfloat startDist, jfloat step, jint n) {

        AutoFocus *method = (AutoFocus*)mp;

	method->mergeFocus(startDist, step, n);

        return as_byte_array(jnienv, method->out->pixel, method->in->width*method->in->height);
}


JNIEXPORT jlong JNICALL Java_Holography_initReconstruct
(JNIEnv *jnienv, jobject obj, jint width, jint height, jbyteArray jimg) {

        unsigned char *pix = as_unsigned_char_array(jnienv, jimg);

        Image *in = new Image(width, height);
	
	in->pixel = pix;

        Method *method = new Method(*in);

        method->WAVELENGTH = 532e-9;

        method->PIXEL_PITCHx = 5.6e-6;
        method->PIXEL_PITCHy = 5.6e-6;
	
	return (jlong)method;
}

JNIEXPORT jlong JNICALL Java_Holography_initAutoFocus
(JNIEnv *jnienv, jobject obj, jint width, jint height, jbyteArray jimg) {

        unsigned char *pix = as_unsigned_char_array(jnienv, jimg);

        Image *in = new Image(width, height);

        in->pixel = pix;

        AutoFocus *method = new AutoFocus(*in,50);

        method->WAVELENGTH = 532e-9;

        method->PIXEL_PITCHx = 5.6e-6;
        method->PIXEL_PITCHy = 5.6e-6;

        return (jlong)method;
}



