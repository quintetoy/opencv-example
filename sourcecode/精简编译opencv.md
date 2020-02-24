### 精简编译opencv

[TOC]

参考博客1:<https://cloud.tencent.com/developer/article/1011652>

需求：只编译opencv_core,opencv_imgproc,opencv_highgui三个库

Opencv_highgui是动态连接系统中的图像编解码库，如果要做静态库，那么也需要将这些解码库静态编译进来。

opencv源码中3rdparty文件夹下包含了这些图像解码库，只要在cmake生成makefile脚本时制定编译这些库。



静态编译以上库的cmake命令

```
cmake . -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX=$install_path \
			-DBZIP2_LIBRARIES=$BZIP2_INSTALL_PATH/lib/libbz2.a\
			-DBUILD_DOCS=off\
			-DBUILD_SHARED_LIBS=off\
			-DBUILD_FAT_JAVA_LIB=off\
			-DBUILD_TESTS=off\
			-DBUILD_TIFF=on\
			-DBUILD_JASPER=on\
			-DBUILD_JPEG=on\
			-DBUILD_OPENEXR=on\
			-DBUILD_PNG=on\
			-DBUILD_ZLIB=on\
			-DBUILD_opencv_apps=off\
			-DBUILD_opencv_calib3d=off\
			-DBUILD_opencv_features2d=off \
    -DBUILD_opencv_flann=off \
    -DBUILD_opencv_gpu=off \
    -DBUILD_opencv_java=off \
    -DBUILD_opencv_legacy=off \
    -DBUILD_opencv_ml=off \
    -DBUILD_opencv_nonfree=off \
    -DBUILD_opencv_objdetect=off \
    -DBUILD_opencv_ocl=off \
    -DBUILD_opencv_photo=off \
    -DBUILD_opencv_python=off \
    -DBUILD_opencv_stitching=off \
    -DBUILD_opencv_superres=off \
    -DBUILD_opencv_ts=off \
    -DBUILD_opencv_video=off \
    -DBUILD_opencv_videostab=off \
    -DBUILD_opencv_world=off \
    -DBUILD_opencv_lengcy=off \
    -DBUILD_opencv_lengcy=off \
    -DWITH_1394=off \
    -DWITH_EIGEN=off \
    -DWITH_FFMPEG=off \
    -DWITH_GIGEAPI=off \
    -DWITH_GSTREAMER=off \
    -DWITH_GTK=off \
    -DWITH_PVAPI=off \
    -DWITH_V4L=off \
    -DWITH_LIBV4L=off \
    -DWITH_CUDA=off \
    -DWITH_CUFFT=off \
    -DWITH_OPENCL=off \
    -DWITH_OPENCLAMDBLAS=off \
    -DWITH_OPENCLAMDFFT=off

make -j 8 install
```



博客2，<https://www.jianshu.com/p/a9d36db94010>

用cmake-gui进行编译





opencv安装全套指南及常见问题解答—on ubuntu

<https://blog.csdn.net/ah_107/article/details/101060094>

运行的命令整理

```
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/home/mmm/opencv02 \ 
      -D INSTALL_C_EXAMPLES=ON \
      -D INSTALL_PYTHON_EXAMPLES=OFF \
      -D WITH_CUDA=ON \
      -D WITH_TBB=ON \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D WITH_OPENGL=ON \
      -D WITH_OPENMP=ON \
      -D BUILD_EXAMPLES=ON ..

```





Android继承opencv并减少库大小

<https://www.jianshu.com/p/697def71d779>



hi3519/hi3516AV200交叉编译opencv的精简版本

<http://www.roselady.vip/a/cangjingge/boke/2018/0714/714.html>





知乎 <https://www.zhihu.com/search?type=content&q=%E5%8F%AA%E7%BC%96%E8%AF%91opencv%E7%9A%84%E9%83%A8%E5%88%86%E5%BA%93>

opencv库的裁剪



##### 编译静态库的命令

```
cmake -D CMAKE_BUILD_TYPE=RELEASE       -D CMAKE_INSTALL_PREFIX=/home/mmm/opencv02       -D BUILD_SHARED_LIBS=NO       -D BUILD_PNG=ON       -D BUILD_JASPER=ON       -D BUILD_JPEG=ON       -D BUILD_TIFF=ON       -D BUILD_ZLIB=ON       -D WITH_JPEG=ON       -D WITH_PNG=ON       -D WITH_JASPER=ON       -D WITH_TIFF=ON ..

make -j8
make install

```



##### 编译动态库的命令

```
cmake -D CMAKE_BUILD_TYPE=RELEASE \ -D CMAKE_INSTALL_PREFIX=/home/mmm/opencv02 \ -D INSTALL_C_EXAMPLES=ON \ -D INSTALL_PYTHON_EXAMPLES=OFF \  -D WITH_CUDA=ON \ -D WITH_TBB=ON \ -D OPENCV_ENABLE_NONFREE=ON \ -D WITH_OPENGL=ON \ -D WITH_OPENMP=ON \ -D BUILD_EXAMPLES=ON \ -DBUILD_opencv_video=off \ -DBUILD_opencv_calib3d=off\ -DBUILD_opencv_features2d=off ..

//未通过验证
```

```
cmake ../ -DCMAKE_BUILD_TYPE=RELEASE -DsCMAKE_INSTALL_PREFIX=/usr/local
```



#####选择控制编译哪些库

```
cmake -D CMAKE_BUILD_TYPE=RELEASE \ -D CMAKE_INSTALL_PREFIX=/home/mmm/opencv02 \ -D INSTALL_C_EXAMPLES=ON
```





参考博客

<https://my.oschina.net/yishanhu/blog/3005155>

```c++
cmake   -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \ //不编译额外的库，所以，这句话删除
-D CMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -std=c++11" \
-D CPACK_GENERATOR=DEB                              \
-D CPACK_BINARY_DEB=ON                              \
-D BUILD_CUDA_STUBS=OFF                             \
-D BUILD_DOCS=OFF                                   \
-D BUILD_EXAMPLES=OFF                               \
-D BUILD_IPP_IW=ON                                  \
-D BUILD_ITT=ON                                     \
-D BUILD_JASPER=OFF                                 \
-D BUILD_JPEG=OFF                                   \
-D BUILD_OPENEXR=OFF                                \
-D BUILD_PACKAGE=ON                                 \
-D BUILD_PERF_TESTS=OFF                             \
-D BUILD_PNG=ON                                     \
-D BUILD_PROTOBUF=ON                                \
-D BUILD_SHARED_LIBS=ON                             \
-D BUILD_TBB=OFF                                    \
-D BUILD_TESTS=OFF                                  \
-D BUILD_TIFF=OFF                                   \
-D BUILD_WITH_DEBUG_INFO=ON                         \
-D BUILD_WITH_DYNAMIC_IPP=OFF                       \
-D BUILD_ZLIB=OFF                                   \
-D BUILD_opencv_apps=OFF                            \
-D BUILD_opencv_calib3d=OFF                         \
-D BUILD_opencv_core=ON                             \
-D BUILD_opencv_cudaarithm=OFF                      \
-D BUILD_opencv_cudabgsegm=OFF                      \
-D BUILD_opencv_cudacodec=OFF                       \
-D BUILD_opencv_cudafeatures2d=OFF                  \
-D BUILD_opencv_cudafilters=OFF                     \
-D BUILD_opencv_cudaimgproc=OFF                     \
-D BUILD_opencv_cudalegacy=OFF                      \
-D BUILD_opencv_cudaobjdetect=OFF                   \
-D BUILD_opencv_cudaoptflow=OFF                     \
-D BUILD_opencv_cudastereo=OFF                      \
-D BUILD_opencv_cudawarping=OFF                     \
-D BUILD_opencv_cudev=OFF                           \
-D BUILD_opencv_dnn=ON                              \
-D BUILD_opencv_features2d=ON                       \
-D BUILD_opencv_flann=OFF                           \
-D BUILD_opencv_highgui=ON                          \
-D BUILD_opencv_imgcodecs=ON                        \
-D BUILD_opencv_imgproc=ON                          \
-D BUILD_opencv_java=OFF                            \
-D BUILD_opencv_js=OFF                              \
-D BUILD_opencv_ml=OFF                              \
-D BUILD_opencv_objdetect=OFF                       \
-D BUILD_opencv_photo=ON                            \
-D BUILD_opencv_python2=OFF                         \
-D BUILD_opencv_shape=OFF                           \
-D BUILD_opencv_stitching=OFF                       \
-D BUILD_opencv_superres=OFF                        \
-D BUILD_opencv_ts=OFF                              \
-D BUILD_opencv_video=OFF                           \
-D BUILD_opencv_videoio=OFF                         \
-D BUILD_opencv_videostab=OFF                       \
-D BUILD_opencv_world=OFF                           \
-D CMAKE_BUILD_TYPE=Release                         \
-D CMAKE_COLOR_MAKEFILE=OFF                         \
-D CMAKE_CONFIGURATION_TYPES=Release                \
-D CMAKE_EXPORT_COMPILE_COMMANDS=OFF                \
-D CMAKE_SKIP_INSTALL_RPATH=NO                      \
-D CMAKE_SKIP_RPATH=NO                              \
-D CPACK_BINARY_IFW=OFF                             \
-D CPACK_BINARY_NSIS=OFF                            \
-D CPACK_BINARY_RPM=OFF                             \
-D CPACK_BINARY_STGZ=OFF                            \
-D CPACK_BINARY_TBZ2=OFF                            \
-D CPACK_BINARY_TGZ=OFF                             \
-D CPACK_BINARY_TXZ=OFF                             \
-D CPACK_BINARY_TZ=OFF                              \
-D CPACK_SOURCE_TBZ2=OFF                            \
-D CPACK_SOURCE_TGZ=OFF                             \
-D CPACK_SOURCE_TXZ=OFF                             \
-D CPACK_SOURCE_TZ=OFF                              \
-D CPACK_SOURCE_ZIP=OFF                             \
-D CPU_BASELINE=SSE3                         	    \
-D CPU_DISPATH=SSE4_1                               \
-D CV_DISABLE_OPTIMIZATION=OFF                      \
-D CV_ENABLE_INTRINSICS=ON                          \
-D CV_TRACE=ON                                      \
-D ENABLE_CCACHE=ON                                 \
-D ENABLE_COVERAGE=OFF                              \
-D ENABLE_CXX11=ON                                  \
-D ENABLE_FAST_MATH=ON                              \
-D ENABLE_GNU_STL_DEBUG=OFF                         \
-D ENABLE_IMPL_COLLECTION=OFF                       \
-D ENABLE_INSTRUMENTATION=OFF                       \
-D ENABLE_NOISY_WARNINGS=OFF                        \
-D ENABLE_OMIT_FRAME_POINTER=ON                     \
-D ENABLE_PRECOMPILED_HEADERS=ON                    \
-D ENABLE_PROFILING=OFF                             \
-D ENABLE_PYLINT=OFF                                \
-D ENABLE_SOLUTION_FOLDERS=OFF                      \
-D INSTALL_CREATE_DISTRIB=OFF                       \
-D INSTALL_C_EXAMPLES=OFF                           \
-D INSTALL_PYTHON_EXAMPLES=OFF                      \
-D INSTALL_TESTS=OFF                                \
-D INSTALL_TO_MANGLED_PATHS=OFF                     \
-D LAPACK_CBLAS_H=cblas.h                           \
-D LAPACK_IMPL:=OpnBLAS                             \
-D LAPACK_INCLUDE_DIR=/usr/include                  \
-D LAPACK_LAPACKE_H=lapacke.h                       \
-D LAPACK_LIBRARIES=/usr/lib/libopenblas.so         \
-D MKL_WITH_OPENMP=OFF                              \
-D MKL_WITH_TBB=OFF                                 \
-D OPENCL_FOUND=ON                                  \
-D OPENCV_ENABLE_NONFREE=OFF                        \
-D OPENCV_FORCE_PYTHON_LIBS=OFF                     \
-D OPENCV_WARNINGS_ARE_ERRORS=OFF                   \
-D PROTOBUF_UPDATE_FILES=OFF                        \
-D WITH_1394=OFF                                    \
-D WITH_ARAVIS=OFF                                  \
-D WITH_CLP=OFF                                     \
-D WITH_CUBLAS=OFF                                  \
-D WITH_CUDA=OFF                                    \
-D WITH_CUFFT=OFF                                   \
-D WITH_EIGEN=ON                                    \
-D WITH_FFMPEG=ON                                   \
-D WITH_GDAL=OFF                                    \
-D WITH_GDCM=OFF                                    \
-D WITH_GIGEAPI=OFF                                 \
-D WITH_GPHOTO2=ON                                  \
-D WITH_GSTREAMER=ON                                \
-D WITH_GSTREAMER_0_10=OFF                          \
-D WITH_GTK=ON                                      \
-D WITH_GTK_2_X=ON                                  \
-D WITH_HALIDE=OFF                                  \
-D WITH_IPP=ON                                      \
-D WITH_ITT=ON                                      \
-D WITH_JASPER=OFF                                  \
-D WITH_JPEG=ON                                     \
-D WITH_LAPACK=ON                                   \
-D WITH_LIBV4L=OFF                                  \
-D WITH_MATLAB=ON                                   \
-D WITH_MFX=OFF                                     \
-D WITH_NVCUVID=ON                                  \
-D WITH_OPENCL=OFF                                  \
-D WITH_OPENCLAMDBLAS=OFF                           \
-D WITH_OPENCLAMDFFT=OFF                            \
-D WITH_OPENCL_SVM=OFF                              \
-D WITH_OPENEXR=OFF                                 \
-D WITH_OPENGL=OFF                                  \
-D WITH_OPENMP=OFF                                  \
-D WITH_OPENNI=OFF                                  \
-D WITH_OPENNI2=OFF                                 \
-D WITH_OPENVX=OFF                                  \
-D WITH_PNG=ON                                      \
-D WITH_PTHREADS_PF=ON                              \
-D WITH_PVAPI=OFF                                   \
-D WITH_QT=OFF                                      \
-D WITH_TBB=OFF                                     \
-D WITH_TIFF=ON                                     \
-D WITH_UNICAP=OFF                                  \
-D WITH_V4L=OFF                                     \
-D WITH_VA=OFF                                      \
-D WITH_VA_INTEL=OFF                                \
-D WITH_VTK=OFF                                     \
-D WITH_WEBP=OFF                                    \
-D WITH_XIMEA=OFF                                   \
-D WITH_XINE=OFF                                    \
-D opencv_dnn_BUILD_TORCH_IMPORTER=ON               \
-D opencv_dnn_PERF_CAFFE=OFF                        \
-D opencv_dnn_PERF_CLCAFFE=OFF                      \
-D BUILD_opencv_freetype=ON \
-D BUILD_opencv_xfeatures2d=OFF  \
-D BUILD_opencv_ximgproc=OFF  \
-D BUILD_opencv_xobjdetect=OFF \
-D BUILD_opencv_xphoto=OFF \
-D BUILD_opencv_reg=OFF \
-D BUILD_opencv_rgbd=OFF \
-D BUILD_opencv_saliency=OFF \
-D BUILD_opencv_shape=OFF \
-D BUILD_opencv_stereo=OFF \
-D BUILD_opencv_stitching=OFF \
-D BUILD_opencv_structured_light=OFF \
-D BUILD_opencv_superres=OFF \
-D BUILD_opencv_surface_matching=OFF \
-D BUILD_opencv_text=OFF \
-D BUILD_opencv_tracking=OFF \
-D BUILD_opencv_ts=OFF \
-D BUILD_opencv_hdf=OFF \
-D BUILD_opencv_plot=OFF \
-D BUILD_opencv_line_descriptor=OFF \
-D BUILD_opencv_fuzzy=OFF \
-D BUILD_opencv_bioinspired=OFF \
-D BUILD_opencv_reg=OFF \
-D BUILD_opencv_saliency=OFF \
-D BUILD_opencv_img_hash=OFF \
 .. 
```



参考博客，最终以这篇为准

<https://blog.csdn.net/xxboy61/article/details/97612723>

opencv3.4.6的一些库

```c++
 calib3d core dnn features2d flann highgui imgcodecs imgproc ml objdetect photo python2 shape stitching superres ts video videoio videostab
 
 --Disabled:world
 
 --     Unavailable:                 cudaarithm cudabgsegm cudacodec cudafeatures2d cudafilters cudaimgproc cudalegacy cudaobjdetect cudaoptflow cudastereo cudawarping cudev java js python3 viz
 
 
 //运行命令 cmake -LA ../ 会产生一系列的检查库等cache的预编译命令
 cmake cache
 //必要的选择模块的命令
-DBUILD_opencv_apps=OFF\
-DBUILD_opencv_calib3d=OFF\
-DBUILD_opencv_core=ON\
-DBUILD_opencv_dnn=OFF\
-DBUILD_opencv_features2d=OFF\
-DBUILD_opencv_flann=OFF\
-DBUILD_opencv_highgui=OFF\
-DBUILD_opencv_imgcodecs=ON\
-DBUILD_opencv_imgproc=ON\
-DBUILD_opencv_java_bindings_generator=OFF\
-DBUILD_opencv_js=OFF\
-DBUILD_opencv_ml=OFF\
-DBUILD_opencv_objdetect=OFF\
-DBUILD_opencv_photo=OFF\
-DBUILD_opencv_python2=OFF\
-DBUILD_opencv_python_bindings_generator=OFF\
-DBUILD_opencv_shape=OFF\
-DBUILD_opencv_stitching=OFF\
-DBUILD_opencv_superres=OFF\
-DBUILD_opencv_ts=OFF\
-DBUILD_opencv_video=OFF\
-DBUILD_opencv_videoio=OFF\
-DBUILD_opencv_videostab=OFF\
-DBUILD_opencv_world=OFF\

```



另外的一些命令记录

```
//依赖库
-DWITH_OPENCL=ON\
-DWITH_CUDA=OFF\
-DWITH_OPENGL=OFF\
-DWITH_OPENMP=OFF\
-DWITH_OPENNI=OFF\
-DWITH_OPENNI2=OFF\
-DWITH_OPENVX=OFF\
-DWITH_OPENEXR=ON\


//其余的代码库，部分被截取的
-DBUILD_CUDA_STUBS=OFF\
-DBUILD_DOCS=OFF\
-DBUILD_EXAMPLES=OFF\
-DBUILD_IPP_IW=ON\
-DBUILD_ITT=ON\
-DBUILD_JASPER=ON\
-DBUILD_JAVA=OFF\
-DBUILD_JPEG=ON\
-DBUILD_OPENEXR=ON\
-DBUILD_PACKAGE=ON\
-DBUILD_PERF_TESTS=OFF\
-DBUILD_PNG=ON\
-DBUILD_PROTOBUF=ON\
-DBUILD_SHARED_LIBS=ON\
-DBUILD_TBB=OFF\
-DBUILD_TESTS=ON\
-DBUILD_TIFF=ON\
-DBUILD_USE_SYMLINKS=OFF\
-DBUILD_WEBP=ON\
-DBUILD_WITH_DEBUG_INFO=OFF\
-DBUILD_WITH_DYNAMIC_IPP=OFF\
-DBUILD_ZLIB:BOOL=ON\


```

整理一份较为完整的，部分的状态不对，少一些未指定，先编译

```
cmake -DCMAKE_BUILD_TYPE=RELEASE \
			-DCMAKE_INSTALL_PREFIX=/home/mmm/opencv02 \
			-DWITH_OPENCL=ON\
      -DWITH_CUDA=OFF\
      -DWITH_OPENGL=OFF\
      -DWITH_OPENMP=OFF\
      -DWITH_OPENNI=OFF\
      -DWITH_OPENNI2=OFF\
      -DWITH_OPENVX=OFF\
      -DWITH_OPENEXR=ON\
      -DBUILD_CUDA_STUBS=OFF\
      -DBUILD_DOCS=OFF\
      -DBUILD_EXAMPLES=OFF\
      -DBUILD_IPP_IW=ON\
      -DBUILD_ITT=ON\
      -DBUILD_JASPER=ON\
      -DBUILD_JAVA=OFF\
      -DBUILD_JPEG=ON\
      -DBUILD_OPENEXR=ON\
      -DBUILD_PACKAGE=ON\
      -DBUILD_PERF_TESTS=OFF\
      -DBUILD_PNG=ON\
      -DBUILD_PROTOBUF=ON\
      -DBUILD_SHARED_LIBS=ON\
      -DBUILD_TBB=OFF\
      -DBUILD_TESTS=ON\
      -DBUILD_TIFF=ON\
      -DBUILD_USE_SYMLINKS=OFF\
      -DBUILD_WEBP=ON\
      -DBUILD_WITH_DEBUG_INFO=OFF\
      -DBUILD_WITH_DYNAMIC_IPP=OFF\
      -DBUILD_ZLIB:BOOL=ON\
      -DBUILD_opencv_apps=OFF\
      -DBUILD_opencv_calib3d=OFF\
      -DBUILD_opencv_core=ON\
      -DBUILD_opencv_dnn=OFF\
      -DBUILD_opencv_features2d=OFF\
      -DBUILD_opencv_flann=OFF\
      -DBUILD_opencv_highgui=OFF\
      -DBUILD_opencv_imgcodecs=ON\
      -DBUILD_opencv_imgproc=ON\
      -DBUILD_opencv_java_bindings_generator=OFF\
      -DBUILD_opencv_js=OFF\
      -DBUILD_opencv_ml=OFF\
      -DBUILD_opencv_objdetect=OFF\
      -DBUILD_opencv_photo=OFF\
      -DBUILD_opencv_python2=OFF\
      -DBUILD_opencv_python_bindings_generator=OFF\
      -DBUILD_opencv_shape=OFF\
      -DBUILD_opencv_stitching=OFF\
      -DBUILD_opencv_superres=OFF\
      -DBUILD_opencv_ts=OFF\
      -DBUILD_opencv_video=OFF\
      -DBUILD_opencv_videoio=OFF\
      -DBUILD_opencv_videostab=OFF\
      -DBUILD_opencv_world=OFF ..
      
      
```

编译成功，运行的命令如下

```
cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/home/mmm/opencv02-DWITH_OPENCL=ON      -DWITH_CUDA=OFF      -DWITH_OPENGL=OFF      -DWITH_OPENMP=OFF      -DWITH_OPENNI=OFF      -DWITH_OPENNI2=OFF      -DWITH_OPENVX=OFF      -DWITH_OPENEXR=ON      -DBUILD_CUDA_STUBS=OFF      -DBUILD_DOCS=OFF      -DBUILD_EXAMPLES=OFF      -DBUILD_IPP_IW=ON      -DBUILD_ITT=ON      -DBUILD_JASPER=ON      -DBUILD_JAVA=OFF      -DBUILD_JPEG=ON      -DBUILD_OPENEXR=ON      -DBUILD_PACKAGE=ON      -DBUILD_PERF_TESTS=OFF      -DBUILD_PNG=ON      -DBUILD_PROTOBUF=ON      -DBUILD_SHARED_LIBS=ON      -DBUILD_TBB=OFF      -DBUILD_TESTS=ON      -DBUILD_TIFF=ON      -DBUILD_USE_SYMLINKS=OFF      -DBUILD_WEBP=ON      -DBUILD_WITH_DEBUG_INFO=OFF      -DBUILD_WITH_DYNAMIC_IPP=OFF      -DBUILD_ZLIB:BOOL=ON      -DBUILD_opencv_apps=OFF      -DBUILD_opencv_calib3d=OFF      -DBUILD_opencv_core=ON      -DBUILD_opencv_dnn=OFF      -DBUILD_opencv_features2d=OFF      -DBUILD_opencv_flann=OFF      -DBUILD_opencv_highgui=OFF      -DBUILD_opencv_imgcodecs=ON      -DBUILD_opencv_imgproc=ON      -DBUILD_opencv_java_bindings_generator=OFF      -DBUILD_opencv_js=OFF      -DBUILD_opencv_ml=OFF      -DBUILD_opencv_objdetect=OFF      -DBUILD_opencv_photo=OFF      -DBUILD_opencv_python2=OFF      -DBUILD_opencv_python_bindings_generator=OFF      -DBUILD_opencv_shape=OFF      -DBUILD_opencv_stitching=OFF      -DBUILD_opencv_superres=OFF      -DBUILD_opencv_ts=OFF      -DBUILD_opencv_video=OFF      -DBUILD_opencv_videoio=OFF      -DBUILD_opencv_videostab=OFF      -DBUILD_opencv_world=OFF ..
```



##### windows编译的参考博客

windows下的参考博客，未验证 <https://blog.csdn.net/jishuqianjin/article/details/93970329>

<https://cloud.tencent.com/developer/article/1509310>



##### opencv库的整理架构与基础架构的关系

最基本的core模块

```sequence
Title:基础模块
Image processing --> core:依赖
High level GUI -- core:
imgcodecs --> core:依赖
imgcodecs --> Image processing:依赖
High level GUI -- video I/O:依赖
video I/O -- core:依赖




```

编译cmake阶段可配置选择的模块有：

```sequence
Title:cmake可选模块,箭头并不指代任何关系
视频分析--2D特征提取:
2D特征提取--机器学习:
机器学习--异构与并行计算:
异构与并行计算--图像拼接:
图像拼接--超分辨:
超分辨--相机矫正与三维重建:
超分辨--分割与匹配:
分割与匹配--dnn:
dnn--三维可视化:
dnn--object detection:
dnn--扩展模块:
```



