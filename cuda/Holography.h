/* DO NOT EDIT THIS FILE - it is machine generated */
//#include <jni.h>
/* Header for class Holography */

#ifndef _Included_Holography
#define _Included_Holography
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     Holography
 * Method:    initCuda
 * Signature: (II[B)J
 */
JNIEXPORT jlong JNICALL Java_Holography_initReconstruct
  (JNIEnv *, jobject, jint, jint, jbyteArray);

JNIEXPORT jlong JNICALL Java_Holography_initAutoFocus
  (JNIEnv *, jobject, jint, jint, jbyteArray);

JNIEXPORT jbyteArray JNICALL Java_Holography_reconstructAt
(JNIEnv *, jobject, jlong, jfloat);

JNIEXPORT jbyteArray JNICALL Java_Holography_zmap
(JNIEnv *, jobject, jlong, jfloat, jfloat, jint);

#ifdef __cplusplus
}
#endif
#endif
