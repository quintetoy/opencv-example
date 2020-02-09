####opencv-imgproc分析

[TOC]

#####模块主要包含的模块

######Image Filter 图像滤波
头文件是指：
```c++
//主要的函数，包括形态学处理
cv::bilateralFilter
cv::blur
cv::boxFilter
cv::buildPyramid
cv::dilate
cv::erode
cv::filter2D
cv::GaussianBlur
cv::getDerivKernels,cv::getGaborKernel,cv::getGaussianKernel,cv::getStructingElement,cv::Laplacian,cv::medianBlur


cv::morphologyDefaultBorderValue, 
cv::morphologyEx,cv::pyrDown, 
cv::pyrMeanShiftFiltering,
cv::pyrUp,cv::Scharr,
cv::sepFilter2D,cv::Sobel,cv::spatialGradient,cv::sqrBoxFilter

```



###### Geometric Image Transformations，几何图像变换

头文件：   

仿射变换，resize等，

```c++

cv::convertMaps，cv::getAffineTransform，
cv::getAffineTransform
cv::getDefaultNewCameraMatrix
cv::getPerspectiveTransform
cv::getPerspectiveTransform
cv::getRectSubPix
cv::getRotationMatrix2D
cv::initUndistortRectifyMap
cv::initWideAngleProjMap
cv::invertAffineTransform
cv::linearPolar
cv::logPolar
cv::remap
cv::resize, cv::undistort , cv::undistortPoints, cv::warpAffine , cv::warpPerspective ,cv::warpPolar

```







###### Miscellaneous Image tansformations 其他图像变换

```c++
void 	cv::adaptiveThreshold (InputArray src, OutputArray dst, double maxValue, int adaptiveMethod, int thresholdType, int blockSize, double C)
 
void 	cv::blendLinear (InputArray src1, InputArray src2, InputArray weights1, InputArray weights2, OutputArray dst)
 
void 	cv::distanceTransform (InputArray src, OutputArray dst, OutputArray labels, int distanceType, int maskSize, int labelType=DIST_LABEL_CCOMP)
 	 
void 	cv::distanceTransform (InputArray src, OutputArray dst, int distanceType, int maskSize, int dstType=CV_32F)
 
int 	cv::floodFill (InputOutputArray image, Point seedPoint, Scalar newVal, Rect *rect=0, Scalar loDiff=Scalar(), Scalar upDiff=Scalar(), int flags=4)
 
int 	cv::floodFill (InputOutputArray image, InputOutputArray mask, Point seedPoint, Scalar newVal, Rect *rect=0, Scalar loDiff=Scalar(), Scalar upDiff=Scalar(), int flags=4)//填充连通域。
 
void 	cv::grabCut (InputArray img, InputOutputArray mask, Rect rect, InputOutputArray bgdModel, InputOutputArray fgdModel, int iterCount, int mode=GC_EVAL)//一种彩色图像分割算法
 
void 	cv::integral (InputArray src, OutputArray sum, int sdepth=-1)//计算积分图像
 
void 	cv::integral (InputArray src, OutputArray sum, OutputArray sqsum, int sdepth=-1, int sqdepth=-1)
 
void 	cv::integral (InputArray src, OutputArray sum, OutputArray sqsum, OutputArray tilted, int sdepth=-1, int sqdepth=-1)
 
double 	cv::threshold (InputArray src, OutputArray dst, double thresh, double maxval, int type)
 
void 	cv::watershed (InputArray image, InputOutputArray markers)//分水岭图像分割
```





###### Drawing Functions

画长方形，多边形，椭圆，圆等相关函数

```c++
cv::fillPoly,cv::fillConvexPoly,cv::line,cv::polylines,cv::rectangle等
```











###### Color Space Conversions

包含大部分枚举变量，基本的颜色空间转换等。函数包含较少，如下所示

```c++


void 	cv::cvtColor (InputArray src, OutputArray dst, int code, int dstCn=0)
 	Converts an image from one color space to another. More...
 
void 	cv::cvtColorTwoPlane (InputArray src1, InputArray src2, OutputArray dst, int code)
 	Converts an image from one color space to another where the source image is stored in two planes. More...
 
void 	cv::demosaicing (InputArray src, OutputArray dst, int code, int dstCn=0)
 	main function for all demosaicing processes More...
```









###### ColorMaps in Opencv

后续接着分析

```c++
void 	cv::applyColorMap (InputArray src, OutputArray dst, int colormap)
 	Applies a GNU Octave/MATLAB equivalent colormap on a given image. More...
 
void 	cv::applyColorMap (InputArray src, OutputArray dst, InputArray userColor)
 	Applies a user colormap on a given image. More...
```





###### planar Subdivison







###### Histograms，直方图







###### Structural Analysis and Shape Descriptors





###### Motion Analysis and Shape Descriptors





######Feature Detection



###### Object Detection



###### C API



###### Hardware Acceleration Layer





