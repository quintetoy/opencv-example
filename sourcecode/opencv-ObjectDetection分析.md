####opencv-ObjectDetection分析

[TOC]

##### 基本类

 include opencv2/objdetect.hpp

```c++
class cv::BaseCascadeClassifier
class cv::CascadeClassifier
struct cv::DetectionROI
struct cv::HOGDescriptor
class cv::QRCodeDetector
class cv::SimilarRects 
```

函数有

```c++
//人脸检测
Ptr< BaseCascadeClassifier::MaskGenerator > 	cv::createFaceDetectionMaskGenerator ()

  //识别QR，二维码等，返回文本结果
bool 	cv::decodeQRCode (InputArray in, InputArray points, std::string &decoded_info, OutputArray straight_qrcode=noArray())
 	Decode QR code in image and return text that is encrypted in QR code. 
 
  //检测并返回最小四方形的区域
bool 	cv::detectQRCode (InputArray in, std::vector< Point > &points, double eps_x=0.2, double eps_y=0.1)
 	Detect QR code in image and return minimum area of quadrangle that describes QR code. 
 
  //集合所有的待检测目标区域
void 	cv::groupRectangles (std::vector< Rect > &rectList, int groupThreshold, double eps=0.2)
 	Groups the object candidate rectangles. 
 
  
void 	cv::groupRectangles (std::vector< Rect > &rectList, std::vector< int > &weights, int groupThreshold, double eps=0.2)
 
  
void 	cv::groupRectangles (std::vector< Rect > &rectList, int groupThreshold, double eps, std::vector< int > *weights, std::vector< double > *levelWeights)
 
  
void 	cv::groupRectangles (std::vector< Rect > &rectList, std::vector< int > &rejectLevels, std::vector< double > &levelWeights, int groupThreshold, double eps=0.2)
 
  
void 	cv::groupRectangles_meanshift (std::vector< Rect > &rectList, std::vector< double > &foundWeights, std::vector< double > &foundScales, double detectThreshold=0.0, Size winDetSize=Size(64, 128))
```

