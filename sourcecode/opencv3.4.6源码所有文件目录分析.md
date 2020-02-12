#### opencv3.4.6源码所有文件目录分析

[TOC]

#### 一级目录

<https://www.cnblogs.com/carlos-vision/p/6443700.html>

.

├── 3rdparty 包含第三方的库，比如视频编解码的ffmpeg，jpg，png，tiff等图片的开源解码库

├── CMakeLists.txt

├── CONTRIBUTING.md

├── LICENSE

├── README.md

├── apps 包含进行haar分类器训练的工具，opencv进行人脸检测便是基于haar分类器

├── cmake 包含生成工程项目时cmake的依赖文件，用于智能搜索第三方库，

├── data 包含opencv库以及范例中用到的额资源文件，haar物体检测的分类器位于haarcascades子文件中

├── doc 包含生成文档所需的源文件以及辅助脚本

├── include 包含入口头文件，有C风格的，但是在被逐步淘汰，也有c++风格的，推荐使用，opencv.hpp

├── modules 包含核心代码，opencv真正的代码都在这个文件夹中

├── platforms 包含交叉编译所需的工具链以及额外的代码，交叉编译指的是在一个操作系统中编译供另一个系统使用的文件

└── samples 范例文件夹



#### modules中的核心模块记录

androidcamera/，仅用于android平台，使得可以通过与其他平台相同的接口来控制android设备的相机。

core/，核心功能模块，定义了基本的数据结构，包括最重要的 Mat 类、XML 读写、opengl三维渲染等。

imgproc/，全称为 image processing，即图像处理。包括图像滤波、集合图像变换、直方图计算、形状描述子等。图像处理是计算机视觉的重要工具。

imgcodec/，负责各种格式的图片的读写，这个模块是从以前的 highgui 中剥离的。

highgui/，高级图形界面及与 QT 框架的整合。

video/，视频分析模块。包括背景提取、光流跟踪、卡尔曼滤波等，做视频监控的读者会经常使用这个模块。

videoio/，负责视频文件的读写，也包括摄像头、Kinect 等的输入。

calib3d/，相机标定以及三维重建。相机标定用于去除相机自身缺陷导致的画面形变，还原真实的场景，确保计算的准确性。三维重建通常用在双目视觉（立体视觉），即两个标定后的摄像头观察同一个场景，通过计算两幅画面中的相关性来估算像素的深度。

features2d/，包含 2D 特征值检测的框架。包含各种特征值检测器及描述子，例如 FAST、MSER、OBRB、BRISK等。各类特征值拥有统一的算法接口，因此在不影响程序逻辑的情况下可以进行替换。

objdetect/，物体检测模块。包括haar分类器、SVM检测器及文字检测。

ml/，全称为 Machine Learning，即机器学习。包括统计模型、K最近邻、支持向量机、决策树、神经网络等经典的机器学习算法。

flann/，用于在多维空间内聚类及搜索的近似算法，做图像检索的读者对它不会陌生。

photo/，计算摄影学。包括图像修补、去噪、HDR成像、非真实感渲染等。如果读者想实现Photoshop的高级功能，那么这个模块必不可少。

stitching/，图像拼接，可用于制作全景图。

nonfree/，受专利保护的算法。包含SIFT和SURF，从功能上来说这两个算法属于features2d模块的，但由于它们都是受专利保护的，想在项目中可能需要专利方的许可。

shape/，形状匹配算法模块。用于描述形状、比较形状。

softcascade/，另一种物体检测算法，Soft Cascade 分类器。包含检测模块和训练模块。

superres/，全称为 Super Resolution，用于增强图像的分辨率。

videostab/，全称为 Video Stabilization，用于解决相机移动时拍摄的视频不够稳定的问题。

viz/，三维可视化模块。可以认为这个模块实现了一个简单的三维可视化引擎，有各种UI控件和键盘、鼠标交互方式。底层实现基于 VTK 这个第三方库。



#### cuda支持的库

这些模块的名称都以 cuda 为开始，cuda 是显卡制造商 NVIDIA 推出的通用计算语言，在cv3中有大量的模块已经被移植到了 cuda 语言。让我们依次看一下：

cuda/，CUDA-加速的计算机视觉算法，包括数据结构 cuda::GpuMat、 基于cuda的相机标定及三维重建等。

cudaarithm/，CUDA-加速的矩阵运算模块。

cudabgsegm/，CUDA-加速的背景分割模块，通常用于视频监控。

cudacodec/，CUDA-加速的视频编码与解码。

cudafeatures2d/，CUDA-加速的特征检测与描述模块，与features2d/模块功能类似。

cudafilters/，CUDA-加速的图像滤波。

cudaimgproc/，CUDA-加速的图像处理算法，包含直方图计算、霍夫变换等。

cudaoptflow/，CUDA-加速的光流检测算法。

cudastereo/，CUDA-加速的立体视觉匹配算法。

cudawarping/，实现了 CUDA-加速的快速图像变换，包括透视变换、旋转、改变尺寸等。

cudaev/，实现 CUDA 版本的核心功能，类似 core/ 模块中的基础算法。





#### 编译完成以后的lib库文件及其大小

4.0K	libopencv_calib3d_pch_dephelp.a

1.7M	libopencv_calib3d.so

1.7M	libopencv_calib3d.so.3.4

1.7M	libopencv_calib3d.so.3.4.6

4.0K	libopencv_core_pch_dephelp.a

19M	libopencv_core.so

19M	libopencv_core.so.3.4

19M	libopencv_core.so.3.4.6

4.0K	libopencv_dnn_pch_dephelp.a

5.1M	libopencv_dnn.so

5.1M	libopencv_dnn.so.3.4

5.1M	libopencv_dnn.so.3.4.6

4.0K	libopencv_features2d_pch_dephelp.a

1000K	libopencv_features2d.so

1000K	libopencv_features2d.so.3.4

1000K	libopencv_features2d.so.3.4.6

112K	libopencv_flann_pch_dephelp.a

452K	libopencv_flann.so

452K	libopencv_flann.so.3.4

452K	libopencv_flann.so.3.4.6

4.0K	libopencv_highgui_pch_dephelp.a

64K	libopencv_highgui.so

64K	libopencv_highgui.so.3.4

64K	libopencv_highgui.so.3.4.6

4.0K	libopencv_imgcodecs_pch_dephelp.a

3.4M	libopencv_imgcodecs.so

3.4M	libopencv_imgcodecs.so.3.4

3.4M	libopencv_imgcodecs.so.3.4.6

4.0K	libopencv_imgproc_pch_dephelp.a

49M	libopencv_imgproc.so

49M	libopencv_imgproc.so.3.4

49M	libopencv_imgproc.so.3.4.6

4.0K	libopencv_ml_pch_dephelp.a

932K	libopencv_ml.so

932K	libopencv_ml.so.3.4

932K	libopencv_ml.so.3.4.6

4.0K	libopencv_objdetect_pch_dephelp.a

560K	libopencv_objdetect.so

560K	libopencv_objdetect.so.3.4

560K	libopencv_objdetect.so.3.4.6

4.0K	libopencv_perf_calib3d_pch_dephelp.a

4.0K	libopencv_perf_core_pch_dephelp.a

4.0K	libopencv_perf_dnn_pch_dephelp.a

4.0K	libopencv_perf_features2d_pch_dephelp.a

4.0K	libopencv_perf_imgcodecs_pch_dephelp.a

4.0K	libopencv_perf_imgproc_pch_dephelp.a

4.0K	libopencv_perf_objdetect_pch_dephelp.a

4.0K	libopencv_perf_photo_pch_dephelp.a

4.0K	libopencv_perf_stitching_pch_dephelp.a

4.0K	libopencv_perf_superres_pch_dephelp.a

4.0K	libopencv_perf_videoio_pch_dephelp.a

4.0K	libopencv_perf_video_pch_dephelp.a

4.0K	libopencv_photo_pch_dephelp.a

1.1M	libopencv_photo.so

1.1M	libopencv_photo.so.3.4

1.1M	libopencv_photo.so.3.4.6

4.0K	libopencv_shape_pch_dephelp.a

284K	libopencv_shape.so

284K	libopencv_shape.so.3.4

284K	libopencv_shape.so.3.4.6

4.0K	libopencv_stitching_pch_dephelp.a

692K	libopencv_stitching.so

692K	libopencv_stitching.so.3.4

692K	libopencv_stitching.so.3.4.6

4.0K	libopencv_superres_pch_dephelp.a

212K	libopencv_superres.so

212K	libopencv_superres.so.3.4

212K	libopencv_superres.so.3.4.6

4.0K	libopencv_test_calib3d_pch_dephelp.a

4.0K	libopencv_test_core_pch_dephelp.a

4.0K	libopencv_test_dnn_pch_dephelp.a

4.0K	libopencv_test_features2d_pch_dephelp.a

112K	libopencv_test_flann_pch_dephelp.a

4.0K	libopencv_test_highgui_pch_dephelp.a

4.0K	libopencv_test_imgcodecs_pch_dephelp.a

4.0K	libopencv_test_imgproc_pch_dephelp.a

4.0K	libopencv_test_ml_pch_dephelp.a

4.0K	libopencv_test_objdetect_pch_dephelp.a

4.0K	libopencv_test_photo_pch_dephelp.a

4.0K	libopencv_test_shape_pch_dephelp.a

4.0K	libopencv_test_stitching_pch_dephelp.a

4.0K	libopencv_test_superres_pch_dephelp.a

4.0K	libopencv_test_videoio_pch_dephelp.a

4.0K	libopencv_test_video_pch_dephelp.a

4.0K	libopencv_test_videostab_pch_dephelp.a

2.1M	libopencv_ts.a

4.0K	libopencv_ts_pch_dephelp.a

4.0K	libopencv_videoio_pch_dephelp.a

268K	libopencv_videoio.so

268K	libopencv_videoio.so.3.4

268K	libopencv_videoio.so.3.4.6

4.0K	libopencv_video_pch_dephelp.a

496K	libopencv_video.so

496K	libopencv_video.so.3.4

496K	libopencv_video.so.3.4.6

4.0K	libopencv_videostab_pch_dephelp.a

412K	libopencv_videostab.so

412K	libopencv_videostab.so.3.4

412K	libopencv_videostab.so.3.4.6





modules的基础类型文件所占的内存大小

19M	libopencv_core.so

3.4M	libopencv_imgcodecs.so

64K	libopencv_highgui.so 一般用来显示图像需要，imshow这种。

49M	libopencv_imgproc.so //这块的图像处理函数过多，占了1/4的内存空间，所以需要将其中重要的相关函数摘出来



