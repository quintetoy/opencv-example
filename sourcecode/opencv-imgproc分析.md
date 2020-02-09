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









###### ColorMaps in Opencv，给灰度图重新上色

后续接着分析，函数较少，如下图所示

```c++
void 	cv::applyColorMap (InputArray src, OutputArray dst, int colormap)
 	Applies a GNU Octave/MATLAB equivalent colormap on a given image. More...
 
void 	cv::applyColorMap (InputArray src, OutputArray dst, InputArray userColor)
 	Applies a user colormap on a given image. More...
```





###### planar Subdivison，平面细分

<https://www.bbsmax.com/A/gGdXGG9v54/>

连接计算机视觉和计算图像学，问题较复杂，后续分析





###### Histograms，直方图

```c++
//反向投影，作用，目标检测
//一幅图像的反向投影利用了其原始图像的直方图，将该直方图作为一张查找表来找对应像素点的位置，即将目标图像像素点的值设置为原始图像直方图上对应的bin值，该bin值代表了目标区域上该像素值出现的概率。从而得到一幅图像的概率值。
//总结来说，将一幅图像的像素值，比如四个灰度变化作为一个bins，那么每个bin的像素个数去替换原始图像的灰度值，相当于聚类，聚类的像素的灰度值就是像素的个数。

void 	cv::calcBackProject (const Mat *images, int nimages, const int *channels, InputArray hist, OutputArray backProject, const float **ranges, double scale=1, bool uniform=true)

 
void 	cv::calcBackProject (const Mat *images, int nimages, const int *channels, const SparseMat &hist, OutputArray backProject, const float **ranges, double scale=1, bool uniform=true)
 
void 	cv::calcBackProject (InputArrayOfArrays images, const std::vector< int > &channels, InputArray hist, OutputArray dst, const std::vector< float > &ranges, double scale)

//计算直方图
void 	cv::calcHist (const Mat *images, int nimages, const int *channels, InputArray mask, OutputArray hist, int dims, const int *histSize, const float **ranges, bool uniform=true, bool accumulate=false)

 
void 	cv::calcHist (const Mat *images, int nimages, const int *channels, InputArray mask, SparseMat &hist, int dims, const int *histSize, const float **ranges, bool uniform=true, bool accumulate=false)
 
void 	cv::calcHist (InputArrayOfArrays images, const std::vector< int > &channels, InputArray mask, OutputArray hist, const std::vector< int > &histSize, const std::vector< float > &ranges, bool accumulate=false)
 
double 	cv::compareHist (InputArray H1, InputArray H2, int method)

 
double 	cv::compareHist (const SparseMat &H1, const SparseMat &H2, int method)
 
  //自适应直方图均衡
Ptr< CLAHE > 	cv::createCLAHE (double clipLimit=40.0, Size tileGridSize=Size(8, 8))
 
 //识别手势，后续单独分析。
 //一种图像相似度性度量方法，在颜色直方图中，由于光线等的变化会引起图像颜色值的漂移，它们会引起颜色值位置的变化，从而导致直方图匹配失效。
 //EMD的思想是求得从一幅图像转化为另一幅图的代价，用直方图来表示就是求得一个直方图转化为另一个直方图的代价，代价越小，越相似。
float 	cv::EMD (InputArray signature1, InputArray signature2, int distType, InputArray cost=noArray(), float *lowerBound=0, OutputArray flow=noArray())
 	
//直方图均衡化
void 	cv::equalizeHist (InputArray src, OutputArray dst)
 	Equalizes the histogram of a grayscale image. More...
 
float 	cv::wrapperEMD (InputArray signature1, InputArray signature2, int distType, InputArray cost=noArray(), Ptr< float > lowerBound=Ptr< float >(), OutputArray flow=noArray())
 
```







###### Structural Analysis and Shape Descriptors，结构分析和形状描述

针对轮廓，边缘的一些结构和形状描述函数，很常用

官方文档中给出的类有：

霍夫变换等

```c++
class  	cv::GeneralizedHough
 	finds arbitrary template in the grayscale image using Generalized Hough Transform 
 
class  	cv::GeneralizedHoughBallard
 	finds arbitrary template in the grayscale image using Generalized Hough Transform 
 
class  	cv::GeneralizedHoughGuil
 	finds arbitrary template in the grayscale image using Generalized Hough Transform 
 
class  	cv::Moments
 	struct returned by cv::moments 
```





```c++
//把一个连续光滑曲线折线化，对图像轮廓点进行多边形拟合
//目前的使用场景是，给定一个轮廓点，希望将其拟合成一个矩形，因为边缘轮廓的点不是连续的，所以给定适当大小的点与点之间的距离，则可以根据这些输入输出一个最符合实际的多边形（矩形）
void 	cv::approxPolyDP (InputArray curve, OutputArray approxCurve, double epsilon, bool closed)
 	Approximates a polygonal curve(s) with the specified precision.

//计算轮廓长度，周长或者曲线长度  
double 	cv::arcLength (InputArray curve, bool closed)
 	Calculates a contour perimeter or a curve length.
 
  //计算外接矩形的大小，计算轮廓的垂直边界最小矩形，矩形与图像上下边界平行
Rect 	cv::boundingRect (InputArray array)
 	Calculates the up-right bounding rectangle of a point set or non-zero pixels of gray-scale image. 
 
  //寻找盒子的顶点
void 	cv::boxPoints (RotatedRect box, OutputArray points)
 	Finds the four vertices of a rotated rect. Useful to draw the rotated rectangle. 
  
 //计算二值图像的连通域标记图像
int 	cv::connectedComponents (InputArray image, OutputArray labels, int connectivity, int ltype, int ccltype)
 	computes the connected components labeled image of boolean image 
 
int 	cv::connectedComponents (InputArray image, OutputArray labels, int connectivity=8, int ltype=CV_32S)
 
  //产生连通域图像，并且输出每个域的统计结果
int 	cv::connectedComponentsWithStats (InputArray image, OutputArray labels, OutputArray stats, OutputArray centroids, int connectivity, int ltype, int ccltype)
 	computes the connected components labeled image of boolean image and also produces a statistics output for each label 
 
int 	cv::connectedComponentsWithStats (InputArray image, OutputArray labels, OutputArray stats, OutputArray centroids, int connectivity=8, int ltype=CV_32S)

    //计算整个轮廓或部分轮廓的面积，
    //目前的使用情况是返回最大（小）的轮廓的所有点集
double 	cv::contourArea (InputArray contour, bool oriented=false)
 	Calculates a contour area. 
 
 //计算凸包   
void 	cv::convexHull (InputArray points, OutputArray hull, bool clockwise=false, bool returnPoints=true)
 	Finds the convex hull of a point set.
 
    //发现轮廓凸形缺陷
void 	cv::convexityDefects (InputArray contour, InputArray convexhull, OutputArray convexityDefects)
 	Finds the convexity defects of a contour. 
 
    
Ptr< GeneralizedHoughBallard > 	cv::createGeneralizedHoughBallard ()
 	Creates a smart pointer to a cv::GeneralizedHoughBallard class and initializes it. 
 
Ptr< GeneralizedHoughGuil > 	cv::createGeneralizedHoughGuil ()
 	Creates a smart pointer to a cv::GeneralizedHoughGuil class and initializes it. 
 
    //找出二值图像的轮廓点
void 	cv::findContours (InputOutputArray image, OutputArrayOfArrays contours, OutputArray hierarchy, int mode, int method, Point offset=Point())
 	Finds contours in a binary image. 
 
void 	cv::findContours (InputOutputArray image, OutputArrayOfArrays contours, int mode, int method, Point offset=Point())
 
    //二维点集椭圆拟合
RotatedRect 	cv::fitEllipse (InputArray points)
 	Fits an ellipse around a set of 2D points. 
 
RotatedRect 	cv::fitEllipseAMS (InputArray points)
 	Fits an ellipse around a set of 2D points. 
 
RotatedRect 	cv::fitEllipseDirect (InputArray points)
 	Fits an ellipse around a set of 2D points. More...

    //点集的直线拟合
void 	cv::fitLine (InputArray points, OutputArray line, int distType, double param, double reps, double aeps)
 	Fits a line to a 2D or 3D point set. 
 
void 	cv::HuMoments (const Moments &moments, double hu[7])
 	Calculates seven Hu invariants. 
 
void 	cv::HuMoments (const Moments &m, OutputArray hu)

    //两个凸多边形的交集
float 	cv::intersectConvexConvex (InputArray _p1, InputArray _p2, OutputArray _p12, bool handleNested=true)
 	finds intersection of two convex polygons 
 
bool 	cv::isContourConvex (InputArray contour)
 	Tests a contour convexity.
 
    //比较两个形状
double 	cv::matchShapes (InputArray contour1, InputArray contour2, int method, double parameter)
 	Compares two shapes. 

    //寻找最小面积的包围矩形，可以旋转
RotatedRect 	cv::minAreaRect (InputArray points)
 	Finds a rotated rectangle of the minimum area enclosing the input 2D point set. 

    //寻找最小面积的包围圆形
void 	cv::minEnclosingCircle (InputArray points, Point2f &center, float &radius)
 	Finds a circle of the minimum area enclosing a 2D point set. 
 
    
double 	cv::minEnclosingTriangle (InputArray points, OutputArray triangle)
 	Finds a triangle of minimum area enclosing a 2D point set and returns its area. 
 
Moments 	cv::moments (InputArray array, bool binaryImage=false)
 	Calculates all of the moments up to the third order of a polygon or rasterized shape. More...
 
    //判断一个点是否在轮廓内
double 	cv::pointPolygonTest (InputArray contour, Point2f pt, bool measureDist)
 	Performs a point-in-contour test. 
 
    //两个旋转矩形的交集
int 	cv::rotatedRectangleIntersection (const RotatedRect &rect1, const RotatedRect &rect2, OutputArray intersectingRegion)
 	Finds out if there is any intersection between two rotated rectangles. 
```





###### Motion Analysis and Object Tracking，运动分析和物体跟踪

```c++
//图像累加
void 	cv::accumulate (InputArray src, InputOutputArray dst, InputArray mask=noArray())
 	Adds an image to the accumulator image. 
 
  //将图像元素相乘的结果加到累加图上
void 	cv::accumulateProduct (InputArray src1, InputArray src2, InputOutputArray dst, InputArray mask=noArray())
 	Adds the per-element product of two input images to the accumulator image. 
 
void 	cv::accumulateSquare (InputArray src, InputOutputArray dst, InputArray mask=noArray())
 	Adds the square of a source image to the accumulator image. 
 
void 	cv::accumulateWeighted (InputArray src, InputOutputArray dst, double alpha, InputArray mask=noArray())
 	Updates a running average. 
 
  //生成汉明窗
void 	cv::createHanningWindow (OutputArray dst, Size winSize, int type)
 	This function computes a Hanning window coefficients in two dimensions. More...
 
  //相位相关，这个函数是用来检测两个函数之间的变换平移
Point2d 	cv::phaseCorrelate (InputArray src1, InputArray src2, InputArray window=noArray(), double *response=0)
 	The function is used to detect translational shifts that occur between two images. 
 
```



######Feature Detection，特征检测

```c++
//canny边缘检测
void 	cv::Canny (InputArray image, OutputArray edges, double threshold1, double threshold2, int apertureSize=3, bool L2gradient=false)
 	Finds edges in an image using the Canny algorithm [33] . 
 
void 	cv::Canny (InputArray dx, InputArray dy, OutputArray edges, double threshold1, double threshold2, bool L2gradient=false)
 
  //计算图像块的特征值和特征向量，用于角点检测
void 	cv::cornerEigenValsAndVecs (InputArray src, OutputArray dst, int blockSize, int ksize, int borderType=BORDER_DEFAULT)
 	Calculates eigenvalues and eigenvectors of image blocks for corner detection. More...

    //计算harris角点
void 	cv::cornerHarris (InputArray src, OutputArray dst, int blockSize, int ksize, double k, int borderType=BORDER_DEFAULT)
 	Harris corner detector. More...
 
    //计算梯度矩阵的最小特征值用于角点检测
void 	cv::cornerMinEigenVal (InputArray src, OutputArray dst, int blockSize, int ksize=3, int borderType=BORDER_DEFAULT)
 	Calculates the minimal eigenvalue of gradient matrices for corner detection. More...
 
    //进一步精确角点的位置，细化处理，亚像素角点检测
void 	cv::cornerSubPix (InputArray image, InputOutputArray corners, Size winSize, Size zeroZone, TermCriteria criteria)
 	Refines the corner locations. 
 
Ptr< LineSegmentDetector > 	cv::createLineSegmentDetector (int _refine=LSD_REFINE_STD, double _scale=0.8, double _sigma_scale=0.6, double _quant=2.0, double _ang_th=22.5, double _log_eps=0, double _density_th=0.7, int _n_bins=1024)
 	Creates a smart pointer to a LineSegmentDetector object and initializes it. More...

    //像素级角点检测
void 	cv::goodFeaturesToTrack (InputArray image, OutputArray corners, int maxCorners, double qualityLevel, double minDistance, InputArray mask=noArray(), int blockSize=3, bool useHarrisDetector=false, double k=0.04)
 	Determines strong corners on an image. 
 
void 	cv::goodFeaturesToTrack (InputArray image, OutputArray corners, int maxCorners, double qualityLevel, double minDistance, InputArray mask, int blockSize, int gradientSize, bool useHarrisDetector=false, double k=0.04)
 
    //hough检测圆
void 	cv::HoughCircles (InputArray image, OutputArray circles, int method, double dp, double minDist, double param1=100, double param2=100, int minRadius=0, int maxRadius=0)
 	Finds circles in a grayscale image using the Hough transform. 

    //hough线段检测
void 	cv::HoughLines (InputArray image, OutputArray lines, double rho, double theta, int threshold, double srn=0, double stn=0, double min_theta=0, double max_theta=CV_PI)
 	Finds lines in a binary image using the standard Hough transform. 

    //依据概率来检测直线
void 	cv::HoughLinesP (InputArray image, OutputArray lines, double rho, double theta, int threshold, double minLineLength=0, double maxLineGap=0)
 	Finds line segments in a binary image using the probabilistic Hough transform. 

    //在一系列点集上找直线
void 	cv::HoughLinesPointSet (InputArray _point, OutputArray _lines, int lines_max, int threshold, double min_rho, double max_rho, double rho_step, double min_theta, double max_theta, double theta_step)
 	Finds lines in a set of points using the standard Hough transform. 
 
    //计算角点检测的特征图
void 	cv::preCornerDetect (InputArray src, OutputArray dst, int ksize, int borderType=BORDER_DEFAULT)
 	Calculates a feature map for corner detection. 
```



###### Object Detection，目标检测

```c++
//比较图像之间的重叠区域
//支持6种匹配方式
//平方差，归一化平方差，相关性匹配，归一化的相关性匹配方法，相关性系数匹配方法，归一化的相关性系数匹配方法
void 	cv::matchTemplate (InputArray image, InputArray templ, OutputArray result, int method, InputArray mask=noArray())
 	Compares a template against overlapped image regions. 
```



###### C API

包含该文件夹下的一些宏，枚举等，前面的功能函数的c接口



###### Hardware Acceleration Layer





