##### opencv源码阅读分析

[TOC]

##### Mat整个结构梳理

<https://www.xuebuyuan.com/2153735.html>

文章中主要的内容有：mat类的结构，内存管理，运算，_InputArray类

主要涉及文件有：

include opencv2 core core.hpp 其中有mat类定义

​					mat.hpp mat类的实现

​                                        types_c.h  C语言类型定义和部分



modules core src arithm.cpp 算法实现

​				matop.cpp mat的运算



###### mat类的结构

opencv中定义了Matx（小矩阵）的结构，Vec是matx的子类。

他们通过_InputArray(输入数组)类实现相互转换。在矩阵运算、图像处理等函数中，通过以_InputArray

类为代理，访问Mat或Vec，Mat_<T>是mat的子类，是mat的模板化。



- Mat的结构定义

```
opencv3.4.6 mat.hpp 2098-2134行

int flags; //长度0-31，由5部分组成
int dims; 
int rows,cols;
uchar* data;//data指针指向了数据，
int* refcount;//指针指向了一个计数器，当为0时，释放data
uchar* datastart;
uchar* dataend;
uchar* datalimit;
Mat Allocator* allocator;//allocator动态内存申请器
MSize size;
MStep step;//保存了一行的宽度
```



- - flag的五部分分别是

    0-2:depth深度，在types_c.h 中

    3-11:channel，通道数减1，是灰度图像

    14:MAT_CONT_FLAG,矩阵的数据存储是否连续的标记，1表示连续，0表示不连续

    15:SUBMAT_FLAG,子矩阵标记，Mat支持从大矩阵中取出子矩阵，数据没有复制。

    16-31:MAGIC_VAL，由于矩阵有很多种类型，_InputArray代理所有的数据输入时，需要依赖这个量来判断数据类型，这个值就表达了该类的类型。它可以的取值为，SparseMat(系数矩阵)，Mat，cvMat，CvMatND,CvSpareMat,CvHistogram,CvMemStorage,CvSeq,CvSet,IPLImage

    

  - Mat的构造函数、赋值运算符、数据类型转换运算符

    ```
    mat.hpp 的801行开始定义mat这个类
    
    构造函数：810-1058
    Mat();
    Mat(int rows, int cols, int type);
    Mat(Size size, int type);
    Mat(int rows, int cols, int type, const Scalar& s);
    Mat(Size size, int type, const Scalar& s);
    Mat(int ndims, const int* sizes, int type);
    Mat(const std::vector<int>& sizes, int type);
    Mat(int ndims, const int* sizes, int type, const Scalar& s);
    Mat(const std::vector<int>& sizes, int type, const Scalar& s);
    Mat(const Mat& m);
    Mat(int rows, int cols, int type, void* data, size_t step=AUTO_STEP);
    Mat(Size size, int type, void* data, size_t step=AUTO_STEP);
    Mat(int ndims, const int* sizes, int type, void* data, const size_t* steps=0);
    Mat(const std::vector<int>& sizes, int type, void* data, const size_t* steps=0);
    Mat(const Mat& m, const Range& rowRange, const Range& colRange=Range::all());
    Mat(const Mat& m, const Rect& roi);
    Mat(const Mat& m, const Range* ranges);
    Mat(const Mat& m, const std::vector<Range>& ranges);
    template<typename _Tp> explicit Mat(const std::vector<_Tp>& vec, bool copyData=false);
    
    赋值运算符：1250-1264
    void assignTo( Mat& m, int type=-1 ) const;
    Mat& operator = (const Scalar& s);
    Mat& setTo(InputArray value, InputArray mask=noArray());
    
    
    数据转换运算：1639-1654
    括号运算，截取子矩阵
    Mat operator()( Range rowRange, Range colRange ) const;
    
    数据转换：由Mat转换成老式结构CvMat、CvMatND、IplImage
    数据转换：由Mat转换成vector<_Tp>、Vec<_Tp,
    n>、Matx<_Tp, m, n>
    
    
    还包含访问mat中的像素点的函数，模板函数等
    ```

    

  

###### mat的内存管理

<http://www.itkeyword.com/doc/1527766930906624x828/opencv>

- 图像头与图像内容

  ```
  Mat A=Mat::zeros(800,600,CV_8UC3);
  Mat B=A;
  
  
  Mat::zeros(800,600,CV_8UC3)是一个静态函数，用于返回一个全部是0的矩阵，该函数返回的MatExpr对象将是一个右值（有内存，但是没有变量指向它），暂且称之为TMP
  
  TMP:图像头：800*600*3*uchar；计数器：指针；图像内容：指针
  
  Mat A=TMP，执行这一行代码，需要经历两个阶段：第一阶段，TMP的类型是MatExpr，此处会执行类型转换函数MatExpr::operator Mat();
  
  第二阶段是Mat A的构造方法，Mat::Mat(const Mat& m),从一个已有的Mat构造新的Mat。该代码完成后，TMP作为右值会被销毁掉。
  
  
  Mat B=A;二中共享同一段内存，计数器为2
  
  ```

  

- 内存申请

  class CV_EXPORTS MatAllocator,它是个抽象类，供用户自定义申请内存的方法，如果用户没有指定内存申请器，那么Mat会使用默认的申请方法。



- Op是函数操作抽象类，它的子类在matop.cpp中，此处的op是MatOp_Initializer函数初始化操作，在MatOp_Initializer::assign函数中调用了Mat::create函数，从而申请了内存。

  

  

  














