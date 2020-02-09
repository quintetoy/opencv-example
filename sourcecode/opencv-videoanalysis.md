####opencv-videoanalysis分析

[TOC]

官方文档给出的modules有：

```c++
Motion Analysis
Object Tracking
C API
```



##### Motion Analysis

对应于opencv3.4.6的module文件夹中的`video`的src文件

主要的类有：

```c++
class cv::BackgroundSubstractor //前景/背景的基础分割类
  
class cv::BackgroundSubtractorKNN //k近邻的前景/背景分割
  
class cv::BackgroundSubtractorMOG2 //基于高斯混合模型的前景/背景分割
  
```



主要的函数有：

```c++
Ptr<BackgroundSubtractorKNN> cv::createBackgroundSubtractorKNN
Ptr<BackgroundSubtractorMOG2> cv::createBackgroundSubtractorMOG2
```



具体的使用方法举例（待补充）



##### Object Tracking

主要的类有：光流法，卡尔曼滤波，均值移位等

```c++
class cv::DenseOpticalFlow
class cv::DualTVL1OpticalFlow
class cv::FarnebackOpticalFlow
  
class cv::KalmanFilter
class cv::SparseOpticalFlow
  
class cv::SparsePyrLKOpticalFlow
```

Opencv/video/tracking.hpp



```c++
int 	cv::buildOpticalFlowPyramid (InputArray img, OutputArrayOfArrays pyramid, Size winSize, int maxLevel, bool withDerivatives=true, int pyrBorder=BORDER_REFLECT_101, int derivBorder=BORDER_CONSTANT, bool tryReuseInputImage=true)
 	Constructs the image pyramid which can be passed to calcOpticalFlowPyrLK. 
 
void 	cv::calcOpticalFlowFarneback (InputArray prev, InputArray next, InputOutputArray flow, double pyr_scale, int levels, int winsize, int iterations, int poly_n, double poly_sigma, int flags)
 	Computes a dense optical flow using the Gunnar Farneback's algorithm. More...
 
void 	cv::calcOpticalFlowPyrLK (InputArray prevImg, InputArray nextImg, InputArray prevPts, InputOutputArray nextPts, OutputArray status, OutputArray err, Size winSize=Size(21, 21), int maxLevel=3, TermCriteria criteria=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01), int flags=0, double minEigThreshold=1e-4)
 	Calculates an optical flow for a sparse feature set using the iterative Lucas-Kanade method with pyramids. More...
 
RotatedRect 	cv::CamShift (InputArray probImage, Rect &window, TermCriteria criteria)
 	Finds an object center, size, and orientation. More...
 
double 	cv::computeECC (InputArray templateImage, InputArray inputImage, InputArray inputMask=noArray())
 	Computes the Enhanced Correlation Coefficient value between two images 
 
Ptr< DualTVL1OpticalFlow > 	cv::createOptFlow_DualTVL1 ()
 	Creates instance of cv::DenseOpticalFlow. More...
 
Mat 	cv::estimateRigidTransform (InputArray src, InputArray dst, bool fullAffine)
 	Computes an optimal affine transformation between two 2D point sets. More...
 
Mat 	cv::estimateRigidTransform (InputArray src, InputArray dst, bool fullAffine, int ransacMaxIters, double ransacGoodRatio, int ransacSize0)
 
double 	cv::findTransformECC (InputArray templateImage, InputArray inputImage, InputOutputArray warpMatrix, int motionType, TermCriteria criteria, InputArray inputMask, int gaussFiltSize)
 	Finds the geometric transform (warp) between two images in terms of the ECC criterion 
 	
double 	cv::findTransformECC (InputArray templateImage, InputArray inputImage, InputOutputArray warpMatrix, int motionType=MOTION_AFFINE, TermCriteria criteria=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 50, 0.001), InputArray inputMask=noArray())
 
int 	cv::meanShift (InputArray probImage, Rect &window, TermCriteria criteria)
 	Finds an object on a back projection image. 
```

