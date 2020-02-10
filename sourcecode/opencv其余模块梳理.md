####opencv其余模块梳理

[TOC]

##### camera calibration and 3D Reconstruction

include <opencv2/calib3d.hpp>

Functions

```c++
//校正摄像头，通过多个视场的图像来得到摄像机的内外参数
double 	cv::calibrateCamera (InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints, Size imageSize, InputOutputArray cameraMatrix, InputOutputArray distCoeffs, OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs, OutputArray stdDeviationsIntrinsics, OutputArray stdDeviationsExtrinsics, OutputArray perViewErrors, int flags=0, TermCriteria criteria=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, DBL_EPSILON))
 	Finds the camera intrinsic and extrinsic parameters from several views of a calibration pattern. 
 
double 	cv::calibrateCamera (InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints, Size imageSize, InputOutputArray cameraMatrix, InputOutputArray distCoeffs, OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs, int flags=0, TermCriteria criteria=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, DBL_EPSILON))
 
void 	cv::calibrateHandEye (InputArrayOfArrays R_gripper2base, InputArrayOfArrays t_gripper2base, InputArrayOfArrays R_target2cam, InputArrayOfArrays t_target2cam, OutputArray R_cam2gripper, OutputArray t_cam2gripper, HandEyeCalibrationMethod method=CALIB_HAND_EYE_TSAI)
 	Computes Hand-Eye calibration: gTc. 
 
    //从摄像机矩阵中计算摄像机的有效值
void 	cv::calibrationMatrixValues (InputArray cameraMatrix, Size imageSize, double apertureWidth, double apertureHeight, double &fovx, double &fovy, double &focalLength, Point2d &principalPoint, double &aspectRatio)
 	Computes useful camera characteristics from the camera matrix. 
 
    //计算旋转和平移矩阵
void 	cv::composeRT (InputArray rvec1, InputArray tvec1, InputArray rvec2, InputArray tvec2, OutputArray rvec3, OutputArray tvec3, OutputArray dr3dr1=noArray(), OutputArray dr3dt1=noArray(), OutputArray dr3dr2=noArray(), OutputArray dr3dt2=noArray(), OutputArray dt3dr1=noArray(), OutputArray dt3dt1=noArray(), OutputArray dt3dr2=noArray(), OutputArray dt3dt2=noArray())
 	Combines two rotation-and-shift transformations. 
 
    //根据一幅图像上的立体点，计算在另一幅图像上对象的epilines
void 	cv::computeCorrespondEpilines (InputArray points, int whichImage, InputArray F, OutputArray lines)
 	For points in an image of a stereo pair, computes the corresponding epilines in the other image. 
 
    //将homogenous的点转换为欧式距离
void 	cv::convertPointsFromHomogeneous (InputArray src, OutputArray dst)
 	Converts points from homogeneous to Euclidean space. 
 
    
void 	cv::convertPointsHomogeneous (InputArray src, OutputArray dst)
 	Converts points to/from homogeneous coordinates. More...
 
void 	cv::convertPointsToHomogeneous (InputArray src, OutputArray dst)
 	Converts points from Euclidean to homogeneous space. More...
 
    //矫正匹配点
void 	cv::correctMatches (InputArray F, InputArray points1, InputArray points2, OutputArray newPoints1, OutputArray newPoints2)
 	Refines coordinates of corresponding points.
 
    //分解本质矩阵
void 	cv::decomposeEssentialMat (InputArray E, OutputArray R1, OutputArray R2, OutputArray t)
 	Decompose an essential matrix to possible rotations and translation. More...
 
int 	cv::decomposeHomographyMat (InputArray H, InputArray K, OutputArrayOfArrays rotations, OutputArrayOfArrays translations, OutputArrayOfArrays normals)
 	Decompose a homography matrix to rotation(s), translation(s) and plane normal(s). More...

    //分解投影矩阵
void 	cv::decomposeProjectionMatrix (InputArray projMatrix, OutputArray cameraMatrix, OutputArray rotMatrix, OutputArray transVect, OutputArray rotMatrixX=noArray(), OutputArray rotMatrixY=noArray(), OutputArray rotMatrixZ=noArray(), OutputArray eulerAngles=noArray())
 	Decomposes a projection matrix into a rotation matrix and a camera matrix. More...
 
    //画出棋盘格的角点
void 	cv::drawChessboardCorners (InputOutputArray image, Size patternSize, InputArray corners, bool patternWasFound)
 	Renders the detected chessboard corners. More...

    //画出世界/物体坐标系
void 	cv::drawFrameAxes (InputOutputArray image, InputArray cameraMatrix, InputArray distCoeffs, InputArray rvec, InputArray tvec, float length, int thickness=3)
 	Draw axes of the world/object coordinate system from pose estimation. More...

    //在2D点集中计算光学仿射变换
cv::Mat 	cv::estimateAffine2D (InputArray from, InputArray to, OutputArray inliers=noArray(), int method=RANSAC, double ransacReprojThreshold=3, size_t maxIters=2000, double confidence=0.99, size_t refineIters=10)
 	Computes an optimal affine transformation between two 2D point sets. More...
 
int 	cv::estimateAffine3D (InputArray src, InputArray dst, OutputArray out, OutputArray inliers, double ransacThreshold=3, double confidence=0.99)
 	Computes an optimal affine transformation between two 3D point sets. More...
 
cv::Mat 	cv::estimateAffinePartial2D (InputArray from, InputArray to, OutputArray inliers=noArray(), int method=RANSAC, double ransacReprojThreshold=3, size_t maxIters=2000, double confidence=0.99, size_t refineIters=10)
 	Computes an optimal limited affine transformation with 4 degrees of freedom between two 2D point sets. More...
 
    //根据其余有效信息，筛选homography 分解器
void 	cv::filterHomographyDecompByVisibleRefpoints (InputArrayOfArrays rotations, InputArrayOfArrays normals, InputArray beforePoints, InputArray afterPoints, OutputArray possibleSolutions, InputArray pointsMask=noArray())
 	Filters homography decompositions based on additional information. More...

    //滤除斑点噪声
void 	cv::filterSpeckles (InputOutputArray img, double newVal, int maxSpeckleSize, double maxDiff, InputOutputArray buf=noArray())
 	Filters off small noise blobs (speckles) in the disparity map. More...
 
    //找到棋盘格上亚像素精度的角点
bool 	cv::find4QuadCornerSubpix (InputArray img, InputOutputArray corners, Size region_size)
 	finds subpixel-accurate positions of the chessboard corners More...
 
bool 	cv::findChessboardCorners (InputArray image, Size patternSize, OutputArray corners, int flags=CALIB_CB_ADAPTIVE_THRESH+CALIB_CB_NORMALIZE_IMAGE)
 	Finds the positions of internal corners of the chessboard. More...
 
bool 	cv::findCirclesGrid (InputArray image, Size patternSize, OutputArray centers, int flags, const Ptr< FeatureDetector > &blobDetector, CirclesGridFinderParameters parameters)
 	Finds centers in the grid of circles. More...
 
bool 	cv::findCirclesGrid (InputArray image, Size patternSize, OutputArray centers, int flags=CALIB_CB_SYMMETRIC_GRID, const Ptr< FeatureDetector > &blobDetector=SimpleBlobDetector::create())
 
bool 	cv::findCirclesGrid2 (InputArray image, Size patternSize, OutputArray centers, int flags, const Ptr< FeatureDetector > &blobDetector, CirclesGridFinderParameters2 parameters)
 
    //根据两幅图像的对应点计算本质矩阵
Mat 	cv::findEssentialMat (InputArray points1, InputArray points2, InputArray cameraMatrix, int method=RANSAC, double prob=0.999, double threshold=1.0, OutputArray mask=noArray())
 	Calculates an essential matrix from the corresponding points in two images. 
 
Mat 	cv::findEssentialMat (InputArray points1, InputArray points2, double focal=1.0, Point2d pp=Point2d(0, 0), int method=RANSAC, double prob=0.999, double threshold=1.0, OutputArray mask=noArray())
 
    //计算基础矩阵
Mat 	cv::findFundamentalMat (InputArray points1, InputArray points2, int method=FM_RANSAC, double ransacReprojThreshold=3., double confidence=0.99, OutputArray mask=noArray())
 	Calculates a fundamental matrix from the corresponding points in two images. More...
 
Mat 	cv::findFundamentalMat (InputArray points1, InputArray points2, OutputArray mask, int method=FM_RANSAC, double ransacReprojThreshold=3., double confidence=0.99)
 
    //两个平面之间的透视变换
Mat 	cv::findHomography (InputArray srcPoints, InputArray dstPoints, int method=0, double ransacReprojThreshold=3, OutputArray mask=noArray(), const int maxIters=2000, const double confidence=0.995)
 	Finds a perspective transformation between two planes. More...
 
Mat 	cv::findHomography (InputArray srcPoints, InputArray dstPoints, OutputArray mask, int method=0, double ransacReprojThreshold=3)
 
Mat 	cv::getOptimalNewCameraMatrix (InputArray cameraMatrix, InputArray distCoeffs, Size imageSize, double alpha, Size newImgSize=Size(), Rect *validPixROI=0, bool centerPrincipalPoint=false)
 	Returns the new camera matrix based on the free scaling parameter. More...
 
Rect 	cv::getValidDisparityROI (Rect roi1, Rect roi2, int minDisparity, int numberOfDisparities, int SADWindowSize)
 	computes valid disparity ROI from the valid ROIs of the rectified images (that are returned by cv::stereoRectify()) More...
 
Mat 	cv::initCameraMatrix2D (InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints, Size imageSize, double aspectRatio=1.0)
 	Finds an initial camera matrix from 3D-2D point correspondences. More...
 
void 	cv::matMulDeriv (InputArray A, InputArray B, OutputArray dABdA, OutputArray dABdB)
 	Computes partial derivatives of the matrix product for each multiplied matrix. More...
 
    //将3D点投影到图像平面上
void 	cv::projectPoints (InputArray objectPoints, InputArray rvec, InputArray tvec, InputArray cameraMatrix, InputArray distCoeffs, OutputArray imagePoints, OutputArray jacobian=noArray(), double aspectRatio=0)
 	Projects 3D points to an image plane. More...
 
int 	cv::recoverPose (InputArray E, InputArray points1, InputArray points2, InputArray cameraMatrix, OutputArray R, OutputArray t, InputOutputArray mask=noArray())
 	Recover relative camera rotation and translation from an estimated essential matrix and the corresponding points in two images, using cheirality check. Returns the number of inliers which pass the check. 
 
int 	cv::recoverPose (InputArray E, InputArray points1, InputArray points2, OutputArray R, OutputArray t, double focal=1.0, Point2d pp=Point2d(0, 0), InputOutputArray mask=noArray())
 
int 	cv::recoverPose (InputArray E, InputArray points1, InputArray points2, InputArray cameraMatrix, OutputArray R, OutputArray t, double distanceThresh, InputOutputArray mask=noArray(), OutputArray triangulatedPoints=noArray())
 
float 	cv::rectify3Collinear (InputArray cameraMatrix1, InputArray distCoeffs1, InputArray cameraMatrix2, InputArray distCoeffs2, InputArray cameraMatrix3, InputArray distCoeffs3, InputArrayOfArrays imgpt1, InputArrayOfArrays imgpt3, Size imageSize, InputArray R12, InputArray T12, InputArray R13, InputArray T13, OutputArray R1, OutputArray R2, OutputArray R3, OutputArray P1, OutputArray P2, OutputArray P3, OutputArray Q, double alpha, Size newImgSize, Rect *roi1, Rect *roi2, int flags)
 	computes the rectification transformations for 3-head camera, where all the heads are on the same line. More...
 
void 	cv::reprojectImageTo3D (InputArray disparity, OutputArray _3dImage, InputArray Q, bool handleMissingValues=false, int ddepth=-1)
 	Reprojects a disparity image to 3D space. More...
 
void 	cv::Rodrigues (InputArray src, OutputArray dst, OutputArray jacobian=noArray())
 	Converts a rotation matrix to a rotation vector or vice versa. More...
 
Vec3d 	cv::RQDecomp3x3 (InputArray src, OutputArray mtxR, OutputArray mtxQ, OutputArray Qx=noArray(), OutputArray Qy=noArray(), OutputArray Qz=noArray())
 	Computes an RQ decomposition of 3x3 matrices. More...
 
double 	cv::sampsonDistance (InputArray pt1, InputArray pt2, InputArray F)
 	Calculates the Sampson Distance between two points. More...
 
int 	cv::solveP3P (InputArray objectPoints, InputArray imagePoints, InputArray cameraMatrix, InputArray distCoeffs, OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs, int flags)
 	Finds an object pose from 3 3D-2D point correspondences. More...
 
bool 	cv::solvePnP (InputArray objectPoints, InputArray imagePoints, InputArray cameraMatrix, InputArray distCoeffs, OutputArray rvec, OutputArray tvec, bool useExtrinsicGuess=false, int flags=SOLVEPNP_ITERATIVE)
 	Finds an object pose from 3D-2D point correspondences. More...
 
bool 	cv::solvePnPRansac (InputArray objectPoints, InputArray imagePoints, InputArray cameraMatrix, InputArray distCoeffs, OutputArray rvec, OutputArray tvec, bool useExtrinsicGuess=false, int iterationsCount=100, float reprojectionError=8.0, double confidence=0.99, OutputArray inliers=noArray(), int flags=SOLVEPNP_ITERATIVE)
 	Finds an object pose from 3D-2D point correspondences using the RANSAC scheme. More...
 
double 	cv::stereoCalibrate (InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints1, InputArrayOfArrays imagePoints2, InputOutputArray cameraMatrix1, InputOutputArray distCoeffs1, InputOutputArray cameraMatrix2, InputOutputArray distCoeffs2, Size imageSize, InputOutputArray R, InputOutputArray T, OutputArray E, OutputArray F, OutputArray perViewErrors, int flags=CALIB_FIX_INTRINSIC, TermCriteria criteria=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 1e-6))
 	Calibrates the stereo camera. More...
 
double 	cv::stereoCalibrate (InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints1, InputArrayOfArrays imagePoints2, InputOutputArray cameraMatrix1, InputOutputArray distCoeffs1, InputOutputArray cameraMatrix2, InputOutputArray distCoeffs2, Size imageSize, OutputArray R, OutputArray T, OutputArray E, OutputArray F, int flags=CALIB_FIX_INTRINSIC, TermCriteria criteria=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 1e-6))
 
void 	cv::stereoRectify (InputArray cameraMatrix1, InputArray distCoeffs1, InputArray cameraMatrix2, InputArray distCoeffs2, Size imageSize, InputArray R, InputArray T, OutputArray R1, OutputArray R2, OutputArray P1, OutputArray P2, OutputArray Q, int flags=CALIB_ZERO_DISPARITY, double alpha=-1, Size newImageSize=Size(), Rect *validPixROI1=0, Rect *validPixROI2=0)
 	Computes rectification transforms for each head of a calibrated stereo camera. More...
 
bool 	cv::stereoRectifyUncalibrated (InputArray points1, InputArray points2, InputArray F, Size imgSize, OutputArray H1, OutputArray H2, double threshold=5)
 	Computes a rectification transform for an uncalibrated stereo camera. More...
 
void 	cv::triangulatePoints (InputArray projMatr1, InputArray projMatr2, InputArray projPoints1, InputArray projPoints2, OutputArray points4D)
 	Reconstructs points by triangulation. More...
 
void 	cv::validateDisparity (InputOutputArray disparity, InputArray cost, int minDisparity, int numberOfDisparities, int disp12MaxDisp=1)
 	validates disparity using the left-right check. The matrix "cost" should be computed by the stereo correspondence algorithm More...
```





##### 2D Features Framework

Include <opencv2/features2d.hpp>

###### Modules

###### #Feature Detection and Description

class

```c++
class cv::AgastFeatureDetector  //Wrapping class for feature detection using the AGAST method
  
  //sift/surf构建金字塔策略使用的是高斯滤波，线性滤波。后续提出了非线形滤波，BFSIFT采取双边滤波与双向匹配，AKAZE作者采用的是非线性扩散滤波
  //主要步骤是AOS构造尺度空间，Hessian矩阵特征点检测，方向指定基于一阶微分图像，描述子生成
  //AKAZE是KAZE的加速版，accelerate KAZE，与sift或surf比较，更加稳定，更快，只有新版的opencv有。
class  	cv::AKAZE
 	Class implementing the AKAZE keypoint detector and descriptor extractor, described in [5]. More...
 
  //binary robust invariant scalable keypoints
  //速度比较：SIFT<SURF<BRISK<FREAK<ORB,主要是利用FAST9-16进行特征点检测
class  	cv::BRISK
 	Class implementing the BRISK keypoint detector and descriptor extractor, described in [114] . More...
 
class  	cv::FastFeatureDetector
 	Wrapping class for feature detection using the FAST method. : More...
 
class  	cv::GFTTDetector
 	Wrapping class for feature detection using the goodFeaturesToTrack function. : More...
 
class  	cv::KAZE
 	Class implementing the KAZE keypoint detector and descriptor extractor, described in [6] . More...
 
  //最大极值稳定区域
  //是一种类似分水岭图像的分割与匹配算法，它具有sift及orb等特征不具备的仿射不变性。近年来广泛用于图像分割与匹配领域。可用于斑点检测，
  //MSER的基本原理是对一幅灰度图像取阈值进行二值化处理。阈值从0-255依次递增。阈值的递增类似于分水岭算法中的水面的上升，随着水面的上升，有一些较矮的丘陵会被淹没。在得到的所有二值图像中，图像中某些连通区域变化很小，甚至没有变化，则该区域就被称为最大稳定极值区域。
class  	cv::MSER
 	Maximally stable extremal region extractor. More...
 
  // Oriented Fast + Rotated BRIEF
class  	cv::ORB
 	Class implementing the ORB (oriented BRIEF) keypoint detector and descriptor extractor. More...
 
  //斑点检测
class  	cv::SimpleBlobDetector
 	Class for extracting blobs from an image. : More...
```



###### #Descriptor Matchers

class

```c++
class  	cv::BFMatcher
 	Brute-force descriptor matcher. 
 
class  	cv::DescriptorMatcher
 	Abstract base class for matching keypoint descriptors. 
 
class  	cv::FlannBasedMatcher
 	Flann-based descriptor matcher. 
```





###### #Drawing Function of Keypoints and Matches

```c++
Classes
struct  	cv::DrawMatchesFlags
 
Functions
void 	cv::drawKeypoints (InputArray image, const std::vector< KeyPoint > &keypoints, InputOutputArray outImage, const Scalar &color=Scalar::all(-1), int flags=DrawMatchesFlags::DEFAULT)
 	Draws keypoints. More...
 
void 	cv::drawMatches (InputArray img1, const std::vector< KeyPoint > &keypoints1, InputArray img2, const std::vector< KeyPoint > &keypoints2, const std::vector< DMatch > &matches1to2, InputOutputArray outImg, const Scalar &matchColor=Scalar::all(-1), const Scalar &singlePointColor=Scalar::all(-1), const std::vector< char > &matchesMask=std::vector< char >(), int flags=DrawMatchesFlags::DEFAULT)
 	Draws the found matches of keypoints from two images. More...
 
void 	cv::drawMatches (InputArray img1, const std::vector< KeyPoint > &keypoints1, InputArray img2, const std::vector< KeyPoint > &keypoints2, const std::vector< std::vector< DMatch > > &matches1to2, InputOutputArray outImg, const Scalar &matchColor=Scalar::all(-1), const Scalar &singlePointColor=Scalar::all(-1), const std::vector< std::vector< char > > &matchesMask=std::vector< std::vector< char > >(), int flags=DrawMatchesFlags::DEFAULT)
```





###### #object categorization

Classes

```c++

//利用词袋方法来计算图像的描述符
class  	cv::BOWImgDescriptorExtractor
 	Class to compute an image descriptor using the bag of visual words. 
 
  //用kmeans来训练词
class  	cv::BOWKMeansTrainer
 	kmeans -based class to train visual vocabulary using the bag of visual words approach. 
 
    //抽象基类
class  	cv::BOWTrainer
 	Abstract base class for training the bag of visual words vocabulary from a set of descriptors. 
```



#####framework for working with different datasets 基于不同数据库的工作框架

include <opencv2/datasets/util.hpp>

######Action Recognition
######Face Recognition
######Gesture Recognition
######Human Pose Estimation
######Image Registration
######Image Segmentation
######Multiview Stereo Matching
######Object Recognition
######Pedestrian Detection
######SLAM
######Text Recognition
######Tracking




#####深度学习的相关模块，机器学习模块，立体视觉模块，图像拼接模块，3D视觉，计算图像学，场景文本检测和识别等







