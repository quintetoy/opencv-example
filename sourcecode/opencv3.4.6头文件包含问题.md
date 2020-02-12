#### opencv3.4.6头文件包含问题

[TOC]

#### 如果不清楚程序所用的函数在哪个文件中声明

那么统一包含 include "opencv2/opencv.hpp”

该头文件中包含了所有的模块的头文件

文件的内容如下：

```c++
#ifndef OPENCV_ALL_HPP
#define OPENCV_ALL_HPP

// File that defines what modules where included during the build of OpenCV
// These are purely the defines of the correct HAVE_OPENCV_modulename values
#include "opencv2/opencv_modules.hpp"

// Then the list of defines is checked to include the correct headers
// Core library is always included --> without no OpenCV functionality available
#include "opencv2/core.hpp"

// Then the optional modules are checked
#ifdef HAVE_OPENCV_CALIB3D
#include "opencv2/calib3d.hpp"
#endif
#ifdef HAVE_OPENCV_FEATURES2D
#include "opencv2/features2d.hpp"
#endif
#ifdef HAVE_OPENCV_DNN
#include "opencv2/dnn.hpp"
#endif
#ifdef HAVE_OPENCV_FLANN
#include "opencv2/flann.hpp"
#endif
#ifdef HAVE_OPENCV_HIGHGUI
#include "opencv2/highgui.hpp"
#endif
#ifdef HAVE_OPENCV_IMGCODECS
#include "opencv2/imgcodecs.hpp"
#endif
#ifdef HAVE_OPENCV_IMGPROC
#include "opencv2/imgproc.hpp"
#endif
#ifdef HAVE_OPENCV_ML
#include "opencv2/ml.hpp"
#endif
#ifdef HAVE_OPENCV_OBJDETECT
#include "opencv2/objdetect.hpp"
#endif
#ifdef HAVE_OPENCV_PHOTO
#include "opencv2/photo.hpp"
#endif
#ifdef HAVE_OPENCV_SHAPE
#include "opencv2/shape.hpp"
#endif
#ifdef HAVE_OPENCV_STITCHING
#include "opencv2/stitching.hpp"
#endif
#ifdef HAVE_OPENCV_SUPERRES
#include "opencv2/superres.hpp"
#endif
#ifdef HAVE_OPENCV_VIDEO
#include "opencv2/video.hpp"
#endif
#ifdef HAVE_OPENCV_VIDEOIO
#include "opencv2/videoio.hpp"
#endif
#ifdef HAVE_OPENCV_VIDEOSTAB
#include "opencv2/videostab.hpp"
#endif
#ifdef HAVE_OPENCV_VIZ
#include "opencv2/viz.hpp"
#endif

// Finally CUDA specific entries are checked and added
#ifdef HAVE_OPENCV_CUDAARITHM
#include "opencv2/cudaarithm.hpp"
#endif
#ifdef HAVE_OPENCV_CUDABGSEGM
#include "opencv2/cudabgsegm.hpp"
#endif
#ifdef HAVE_OPENCV_CUDACODEC
#include "opencv2/cudacodec.hpp"
#endif
#ifdef HAVE_OPENCV_CUDAFEATURES2D
#include "opencv2/cudafeatures2d.hpp"
#endif
#ifdef HAVE_OPENCV_CUDAFILTERS
#include "opencv2/cudafilters.hpp"
#endif
#ifdef HAVE_OPENCV_CUDAIMGPROC
#include "opencv2/cudaimgproc.hpp"
#endif
#ifdef HAVE_OPENCV_CUDAOBJDETECT
#include "opencv2/cudaobjdetect.hpp"
#endif
#ifdef HAVE_OPENCV_CUDAOPTFLOW
#include "opencv2/cudaoptflow.hpp"
#endif
#ifdef HAVE_OPENCV_CUDASTEREO
#include "opencv2/cudastereo.hpp"
#endif
#ifdef HAVE_OPENCV_CUDAWARPING
#include "opencv2/cudawarping.hpp"
#endif

#endif
```







#### 如果明确函数所声明处的文件

则只需包含该模块的头文件即可，头文件的包含一般可以从opencv.hpp中查询到。

但是如果这种情况一定要包含 `opencv2/core.hpp`，没有这个头文件，所有opencv的功能函数都无法使用。

