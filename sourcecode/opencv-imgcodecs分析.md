####opencv-imgcodecs分析

[TOC]

##### Image file reading and writing

官方文档的函数只是一部分，实际文件夹中的文件命名并不是函数名，见后续分析。

官方文档给出的主要函数有以下：

```c++
Mat 	cv::imdecode (InputArray buf, int flags)
 	Reads an image from a buffer in memory. 
 
Mat 	cv::imdecode (InputArray buf, int flags, Mat *dst)
 
bool 	cv::imencode (const String &ext, InputArray img, std::vector< uchar > &buf, const std::vector< int > &params=std::vector< int >())
 	Encodes an image into a memory buffer. More...
 
Mat 	cv::imread (const String &filename, int flags=IMREAD_COLOR)
 	Loads an image from a file. More...
 
bool 	cv::imreadmulti (const String &filename, std::vector< Mat > &mats, int flags=IMREAD_ANYCOLOR)
 	Loads a multi-page image from a file. 
 
bool 	cv::imwrite (const String &filename, InputArray img, const std::vector< int > &params=std::vector< int >())
 	Saves an image to a specified file. 
```





##### 实际文件夹中的文件

├── bitstrm.cpp

├── bitstrm.hpp

├── exif.cpp

├── exif.hpp

├── grfmt_base.cpp

├── grfmt_base.hpp 包含类class BaseImageDecoder 和class BaseImageEncoder

├── grfmt_bmp.cpp

├── grfmt_bmp.hpp 继承基类的BmpDecoder. BmpEncoder

├── grfmt_exr.cpp

├── grfmt_exr.hpp 继承基类的ExrDecoder，ExrEncoder

├── grfmt_gdal.cpp

├── grfmt_gdal.hpp 和以上相似，gdal是一种像素形式

├── grfmt_gdcm.cpp

├── grfmt_gdcm.hpp 同以上，DICOMDecoder

├── grfmt_hdr.cpp

├── grfmt_hdr.hpp 同上HdrDecoder

├── grfmt_jpeg.cpp

├── grfmt_jpeg.hpp

├── grfmt_jpeg2000.cpp

├── grfmt_jpeg2000.hpp

├── grfmt_pam.cpp

├── grfmt_pam.hpp

├── grfmt_png.cpp

├── grfmt_png.hpp

├── grfmt_pxm.cpp

├── grfmt_pxm.hpp

├── grfmt_sunras.cpp

├── grfmt_sunras.hpp

├── grfmt_tiff.cpp

├── grfmt_tiff.hpp

├── grfmt_webp.cpp

├── grfmt_webp.hpp

├── grfmts.hpp

├── ios_conversions.mm

├── loadsave.cpp 最后封装的API，如imdecode等

├── precomp.hpp

├── rgbe.cpp

├── rgbe.hpp RGBE文件Radiance RGB Map Data

├── utils.cpp

└── utils.hpp



##### 分析基类中图像编解码的源码

基本只是定义了一部分的虚拟功能函数，还没有涉及到最后封装的API

```c++
namespace cv{
	class BaseImageDecoder;
	class BaseImageEncoder;
  
  typedef Ptr<BaseImageEncoder> ImageEncoder;
  typedef Ptr<BaseImageDecoder> ImageDecoder;
  
  //图像解码，将编码图像string，vector？解码为mat形式，
  class BaseImageDecoder{
  public:
    BaseImageDecoder();
    virtual ~BaseImageDecoder();
    
    int width() const{return m_width;}
    int height() const{return m_height;}
    virtual int type()const{return m_type;}
    
    virtual bool setSource(const String& filename);
    virtual bool setSource(const Mat& buf);
    
    virtual int setScale(const int& scale_denom);
    
    virtual bool readHeader()=0;
    virtual bool readData(Mat& img)=0;
    
    virtual bool nextPage(){return false;}
    
    virtual size_t signatureLength()const;
    
    virtual bool checkSignature(const String& signature)const;
    virtual ImageDecoder newDecoder()const;
    
   protected:
    int m_width;
    int m_height;
    int m_type;
    int m_scale_denom;
    String m_filename;
    String m_signature;
    Mat m_buf;
    bool m_buf_supported;
  };
  
  class BaseImageEncoder{
  public:
    BaseImageEncoder();
    virtual ~BaseImageEncoder(){}
    virtual bool isFormatSupported(int depth)const;
    
    virtual bool setDestination(const String& filename);
    virtual bool setDestination(std::vector<uchar>& buf);
    virtual bool write(const Mat& img,const std::vector<int>& params);
    virtual bool writemulti(const std::vector<Mat>& img_vec,const std::vector<int>& params);
    
    virtual String getDescription() const;
    virtual ImageEncoder newEncoder() const;
    virtual void throwOnEror()const;
     
  protected:
    String m_description;
    
    String m_filename;
    std::vector<uchar>* m_buf;
    bool m_buf_supported;
    
    String m_last_error;   
  };
}
```



具体的实现方法

```c++
namespace cv{
  BaseImageDecoder::BaseImageDecoder(){
    m_width=m_height=0;
    m_type=-1;
    m_buf_supported=false;
    m_scale_denom=1;
    
  }
  
  bool BaseImageDecoder::setSource( const String& filename )
{
    m_filename = filename;
    m_buf.release();//申请的没有赋值的成员变量。释放这段内存？
    return true;
}

bool BaseImageDecoder::setSource( const Mat& buf )
{
    if( !m_buf_supported )
        return false;
    m_filename = String();
    m_buf = buf;//指向输入的mat那段内存
    return true;
}
  
  size_t BaseImageDecoder::signatureLength() const
{
    return m_signature.size();
}

  bool BaseImageDecoder::checkSignature( const String& signature ) const
{
    size_t len = signatureLength();
    return signature.size() >= len && memcmp( signature.c_str(), m_signature.c_str(), len ) == 0;
}
  
  int BaseImageDecoder::setScale( const int& scale_denom ){
    int temp=m_scale_denom;
    m_scale_denom=scale_denom;
    return temp;//注释，m_scale_denom，初始的结果是1，所以函数返回的结果是1
  }
  
  ImageDecoder BaseImageDecoder::newDecoder() const
{
    return ImageDecoder();//相当于返回的是Ptr<BaseImageDecoder>
}
  
  
  
  
  ////encoder
  BaseImageEncoder::BaseImageEncoder()
{
    m_buf = 0;
    m_buf_supported = false;
}

bool  BaseImageEncoder::isFormatSupported( int depth ) const
{
    return depth == CV_8U;
}

String BaseImageEncoder::getDescription() const
{
    return m_description;
}

bool BaseImageEncoder::setDestination( const String& filename )
{
    m_filename = filename;
    m_buf = 0;
    return true;
}

bool BaseImageEncoder::setDestination( std::vector<uchar>& buf )
{
    if( !m_buf_supported )
        return false;
    m_buf = &buf;
    m_buf->clear();
    m_filename = String();
    return true;
}

bool BaseImageEncoder::writemulti(const std::vector<Mat>&, const std::vector<int>& )
{
    return false;
}

ImageEncoder BaseImageEncoder::newEncoder() const
{
    return ImageEncoder();
}

void BaseImageEncoder::throwOnEror() const
{
    if(!m_last_error.empty())
    {
        String msg = "Raw image encoder error: " + m_last_error;
        CV_Error( CV_BadImageSize, msg.c_str() );
    }
}
  
}
```



imread函数，首先判断文件能否读，是否合法，然后判断图像的编码类型，申请不同类型的解码器，同时，根据hdrtype用不同申请函数申请mat类型的结构来存储图像数据，较为关键的函数decoder->readData

```c++
static void* imread_(const String& filename,int flags,int hdrtype,Mat* mat=0){
  CV_Assert(mat||hdrtype!=LOAD_MAT);
  
  IplImage* image=0;
  CvMat *matrix=0;
  Mat temp,*data=&temp;//二维数组
  
  ImageDecoder decoder;
  
  #ifdef HAVE_GDAL
    if(flags != IMREAD_UNCHANGED && (flags & IMREAD_LOAD_GDAL) == IMREAD_LOAD_GDAL ){
        decoder = GdalDecoder().newDecoder();
    }else{
	#endif
        decoder = findDecoder( filename );
	#ifdef HAVE_GDAL
    }
	#endif
  
  if(!decoder){
    return 0;
  }
  
      int scale_denom = 1;
    if( flags > IMREAD_LOAD_GDAL )
    {
    if( flags & IMREAD_REDUCED_GRAYSCALE_2 )
        scale_denom = 2;
    else if( flags & IMREAD_REDUCED_GRAYSCALE_4 )
        scale_denom = 4;
    else if( flags & IMREAD_REDUCED_GRAYSCALE_8 )
        scale_denom = 8;
    }
  
  decoder->setScale(scale_denom);
  
  decoder->setSource(filename);
  try{
    if(!decoder->readHeader())
      return 0;   
  }
     catch (const cv::Exception& e)
    {
        std::cerr << "imread_('" << filename << "'): can't read header: " << e.what() << std::endl << std::flush;
        return 0;
    }
    catch (...)
    {
        std::cerr << "imread_('" << filename << "'): can't read header: unknown exception" << std::endl << std::flush;
        return 0;
    }
  
  Size size=validateInoutImageSize(Size(decoder->width(),decoder->height()));
  
  int type=decoder->type();
  
      if( (flags & IMREAD_LOAD_GDAL) != IMREAD_LOAD_GDAL && flags != IMREAD_UNCHANGED )
    {
        if( (flags & CV_LOAD_IMAGE_ANYDEPTH) == 0 )
            type = CV_MAKETYPE(CV_8U, CV_MAT_CN(type));

        if( (flags & CV_LOAD_IMAGE_COLOR) != 0 ||
           ((flags & CV_LOAD_IMAGE_ANYCOLOR) != 0 && CV_MAT_CN(type) > 1) )
            type = CV_MAKETYPE(CV_MAT_DEPTH(type), 3);
        else
            type = CV_MAKETYPE(CV_MAT_DEPTH(type), 1);
    }
//常见mat，申请内存
    if( hdrtype == LOAD_CVMAT || hdrtype == LOAD_MAT )
    {
        if( hdrtype == LOAD_CVMAT )
        {
          
            matrix = cvCreateMat( size.height, size.width, type );
            temp = cvarrToMat( matrix );
        }
        else
        {
            mat->create( size.height, size.width, type );
            data = mat;
        }
    }
    else
    {
        image = cvCreateImage(cvSize(size), cvIplDepth(type), CV_MAT_CN(type));
        temp = cvarrToMat( image );
    }

    // read the image data，读取文件的图像数据
    bool success = false;
    try
    {
        if (decoder->readData(*data))
            success = true;
    }
    catch (const cv::Exception& e)
    {
        std::cerr << "imread_('" << filename << "'): can't read data: " << e.what() << std::endl << std::flush;
    }
    catch (...)
    {
        std::cerr << "imread_('" << filename << "'): can't read data: unknown exception" << std::endl << std::flush;
    }
  //释放内存
    if (!success)
    {
        cvReleaseImage( &image );
        cvReleaseMat( &matrix );
        if( mat )
            mat->release();
        return 0;
    }

    if( decoder->setScale( scale_denom ) > 1 ) // if decoder is JpegDecoder then decoder->setScale always returns 1
    {
        resize( *mat, *mat, Size( size.width / scale_denom, size.height / scale_denom ), 0, 0, INTER_LINEAR_EXACT);
    }

    return hdrtype == LOAD_CVMAT ? (void*)matrix :
        hdrtype == LOAD_IMAGE ? (void*)image : (void*)mat;
}
```

