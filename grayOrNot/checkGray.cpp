#include <iostream>
#include <opencv2/opencv.hpp>
#include <string.h>
using namespace std;
using namespace cv;

//2020/1/2
void checkGray(Mat img){
    vector<Mat> channels;
    Mat channels_f[3];
    split(img, channels);
    for(int i=0;i<3;i++)
        //转换为float类型
        channels[i].convertTo(channels_f[i], CV_32F);
    
    
    int s_w=img.cols;
    int s_h=img.rows;
    

    //将mat矩阵的数值转换为float型
    Mat average=(channels_f[0]+channels_f[1]+channels_f[2])/3;
    Mat b_s=abs(channels_f[0]-average);
    Mat g_s=abs(channels_f[1]-average);
    Mat r_s=abs(channels_f[2]-average);
    
    
    
    Scalar b_sum=sum(b_s)/(s_w*s_h);
    Scalar g_sum=sum(g_s)/(s_w*s_h);
    Scalar r_sum=sum(r_s)/(s_w*s_h);
    
    
    Scalar gray_degree=(b_sum+g_sum+r_sum)/3;
    
    cout<<"gray degree is "<<gray_degree[0]<<endl;
    
}




