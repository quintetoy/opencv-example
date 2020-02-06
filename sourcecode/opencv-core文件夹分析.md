##### opencv-core阅读分析

[TOC]

###### core中的基本数据类型和架构

所有的总结均只来自书籍learning opencv

总结来说，core文件夹中包含数据结构，矩阵运算，数据变换，对象持久，内存管理，错误处理，动态装载，绘图，文本和基本的数学功能。

具体的，core文件夹中主要是opencv的一些基本数据类型和操作，包括：

1、mat, point,rect,scalar，cvSIze等

2、包含很多矩阵mat的基本操作，矩阵之间的运算，矩阵的转换，矩阵之间的特征值计算，矩阵求逆，矩阵图像融合，计算矩阵中的非零元素的个数，矩阵的最大最小值元素，矩阵的区域截取等等。基本的矩阵分析中涉及的基本运算均包含，如SVD，PCA等。

3、包含绘图操作，画直线，圆形，椭圆，多边形等（cvFillPoly）

4、字体文字操作等

5、xml等数据存储

6、内存管理：主要是alloc.cpp

关于mat的分析，参考另一篇文档




###### 关于内存管理的相关源码分析
所有的内存空间都是通过自定义的函数cvAlloc()和cvFree()来完成的.
其中的主要函数有以下：
```c++
namespace cv{
//判断是否还有内存可以被分配，如果没有，直接报错
static void* OutOfMemoryError(size_t size){
  CV_Error_(CV_StsNoMem,("Failed to allocate %llu bytes",(unsigned long long )size));
}

void* fasMalloc(size_t size){
  #ifdef HAVE_POSIX_MEMALIGN
  	void* ptr=NULL;
  	if(posix_memalign(&ptr,CV_MALLOC_ALIGN,size))
  		ptr=NULL;
  	if(!ptr)
  		return OutOfMemoryError(size);
  	return ptr;
  #elif defined HAVE_MEMALIGN
  	void* ptr=memalign(CV_MALLOC_ALIGN,size);
  	if(!ptr)
  		return OutOfMemoryError(size);
  	return ptr;
  #else
  	uchar* udata=(uchar*)malloc(size+sizeof(void*)+CV_MALLOC_ALIGN);//多申请了两块内存 sizeof(void*)+CV_MALLOC_ALIGN
  	if(!udata)
  		return OutOfMemoryError(size);
  	//通过alignPtr()函数对齐指针，值得注意的是，函数跳过了一个指针空间
  	//CV_MALLOC_ALIGN，这个常量为16。将申请内存的指针按16位进行了对齐，EE只支持16位对其的指针
  	//返回第一个大于输入指针且最后四位为0的值，这个值在ptr~ptr+15，跳一个指针的值，void*，adata为CV_MALLOC_ALIGN+size，CV_MALLOC_ALIGN这块内存可对应的mat中的flag的属性中的存储值，存储一些mat的属性
  	uchar** adata=alignPtr((uchar**)udata+1,CV_MALLOC_ALIGN);
  	
  	//保存原始的内存地址，adata是对齐后的地址，若直接释放这个地址，则会产生内存泄漏，malloc得到的指针地址保存在adata的前一块区域内。
  	adata[-1]=udata;
  	
  	return adata;
	#endif
}

void fastFree(void* ptr){
#if defined HAVE_POSIX_MEMALIGN || defined HAVE_MEMALIGN //系统自己分配
    free(ptr);
#else
    if(ptr)
    {
    //获取起始指针的值
        uchar* udata = ((uchar**)ptr)[-1];//posix_memalign，预对齐内存的分配
        
        //检查起始指针的有效性，即ptr的值应该在uData与(uData+sizeof(void*)+CV_MALLOC_ALIGN)之间
        CV_DbgAssert(udata < (uchar*)ptr &&
               ((uchar*)ptr - udata) <= (ptrdiff_t)(sizeof(void*)+CV_MALLOC_ALIGN));
        free(udata);
    }
#endif
}
}


CV_IMPL void* cvAlloc(size_t size){//CV_IMPL是opencv定义的宏，意思是编译方式设置为c方式
  return cv::fastMalloc(size);
}

CV_IMPL void cvFree_(void* ptr){
  cv::fastFree(ptr);
}



######内存对齐
内存是由chip构成，每个chip内部，是由8个bank组成的，每个bank内部，就是电容的行列矩阵结构。二维矩阵中的一个元素一般存储着8个bit，也就是包含了8个小电容。
8个同位置的元素，一起组成在内存中连续的64个bit。多层bank并列，然后同一层的多种并列构成连续的内存。


所以，内存对齐最最底层的原因是内存的IO是以64bit为单位进行的。对于64位数据宽度的内存，假如cpu也是64位的cpu，每次内存IO获取数据都是从同行同列的8个chip中各自读取一个字节拼起来的。0-63，一次，64-127一次。

链接：https://www.jianshu.com/p/37409be16a37

假如对于一个c的程序员，如果把一个bigint（64位）地址写到的0x0001开始，而不是0x0000开始，那么数据并没有存在同一行列地址上。因此cpu必须得让内存工作两次才能取到完整的数据。效率自然就很低。




```



###### 后续补齐内存对齐的相关知识

