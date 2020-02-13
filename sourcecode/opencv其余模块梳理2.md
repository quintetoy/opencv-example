###opencv其余模块梳理2

[TOC]

### Improved Background-Foreground Segmentation Methods

该模块下面主要包含的类有下面几种

Include opencv2/bgsegm.hpp

参考博客<https://blog.csdn.net/Anderson_Y/article/details/82082095>

LBP：local binary pattern 中间点与周边的值进行大小判断，大于周边，周边值为0，小于，周边值为1，从左上角开始顺时针描述为 一串二值编码串，类似10000100，三邻域结果是8位长



####class cv::bgsegm::BackgroundSubtractorCNT

cnt算法是一种不需要对背景点进行高斯建模处理的方法。他仅仅只使用过去连续N帧内的像素点值的信息以及其他一点额外的信息，因此速度很快。函数可以设置是否记录历史值。

1、不记录历史值时，通过简单的阈值操作判断像素点的稳定性，如果在连续的minPixelStability帧内都保持稳定，则认为该像素点是稳定的，否则不稳定，稳定的是背景点。

可以记录物体作为前景点的运动轨迹

2、记录历史值，程序在运行过程会记录从程序运行到当前时刻为止，稳定时间最长的像素点的灰度值historyColorRef和它的稳定时间histStabilityRef。在新的一帧到来时，通过一系列的阈值比较操作来判断像素点的稳定性，从而判断是否为背景点。

检测区域更加准确。

####class  cv::bgsegm::BackgroundSubtractorGMG

GMG是背景减除算法中的一种，该算法不是用Gaussian函数对背景进行建模，而是通过像素点的颜色特征对背景进行建模，同时通过贝叶斯公式计算像素点作为背景点（前景点）的极大似然估计，得到一张概率图，通过对概率图的阈值操作（和开闭运算等），来得到前景点和背景点的划分。



####class cv::bgsegm::BackgroundSubtractorGSOC

从博客的参考文章来看，此算法的效果很好，阅读源码分析



####class cv::bgsegm::BackgroundSubtractorLSBP

Background Substraction using local SVD Binary Pattern 2016

LSBP是一种结合LBP特征和SVD的背景减除法。在假设被检测物体为朗伯体的前提下，LSBP特征具有光照不变性。

首先将像素及其邻域内的像素看做一个矩阵，对该矩阵进行SVD分解，得到一系列特征值，求和变为新的像素值

然后提取LSBP特征

具体操作时，提取LSBP特征和颜色特征。对于每一个像素点，计算其与背景模型中的相匹配的模型的个数。



####class cv::bgsegm::BackgroundSubtractorLSBPDesc



####class cv::bgsegm::BackgroundSubtractorMOG

An Improved Adaptive Background Mixture Model for Realtime Tracking with Shadow Detection

基于混合高斯背景建模的背景减除算法，算法对每个像素点使用固定数量的高斯函数来对其像素值的分布进行建模。通过训练得到每个高斯函数的参数，当新的一帧来到时，对场景中的每一个像素计算在N时刻像素值为xn的概率。

判定准则：如果与B个模型的任意一个距离在2.5个标准差以上，前景点

对于与中心的距离在2.5个标准差以内的第一个匹配上的高斯函数，更新。

如果K个都没有匹配上，那么用一个新的高斯函数去替代值最小的那个高斯函数



####class cv::bgsegm::SyntheticSequenceGenerator



#####  *BackgroundSubtractorKNN*



*基类*

```c++
class cv::BackgroundSubstractor{
Public:
  virtual ~BackgroundSubstractor(){}
  virtual void apply(InputArray image,OutputArray fmask,double learningRate=-1);
  virtual void getBackgroundImage(OutputArray image)const;
};

//只需要建立一个基类的指针，通过创建不同的实例方法，即可使用不同的方法进行背景提取
Ptr<BackgroundSubstractor>pBgSub;
pBgSub=createBackgroundSubtractorKNN();
pBgSub->apply(frame,mask);
```




















