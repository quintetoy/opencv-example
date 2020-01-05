#include <vector>
#include <iostream>
#include <fstream>
#include <dirent.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

//中心点,如果是c++的rect，则要换成左上角的点的输入
float overlap(float x1,float w1,float x2,float w2){
    float l1=x1;
    float l2=x2;
    float left=l1>l2?l1:l2;
    float r1=x1+w1;
    float r2=x2+w2;
    float right=r1<r2?r1:r2;
    return right-left;

}


float box_intersection(Rect a,Rect b){
    float w=overlap(a.x,a.width,b.x,b.width);
    float h=overlap(a.y,a.height,b.y,b.height);
    if(w<0||h<0) return 0;
    float area=w*h;
    return area;

}

float box_union(Rect a,Rect b){
    float i=box_intersection(a,b);
    float u=a.width*a.height+b.width*b.height-i;
    return u;
}

float box_iou(Rect a,Rect b){
    return box_intersection(a,b)/box_union(a,b);
}

//修改  两两之间求iou，不符合要求的删除
vector<Rect> do_nms(vector<Rect> boxes,float thresh){
    int i,j;
    vector<Rect> res;
    int total=boxes.size();
    int probs[total];
    memset(probs, 0, total*sizeof(int));
    for(i=0;i<total;++i){
        for(j=i+1;j<total;++j){
            float result=box_iou(boxes[i],boxes[j]);
        
            if(box_iou(boxes[i],boxes[j])>thresh){
                if(probs[i]==0){
                    probs[i]=1;
                    probs[j]=1;
                    res.push_back(boxes[i]);
                }
            }
        }
    }
    cout<<"res size is"<<res.size()<<endl;
    
    return res;
}



void getClusterRect(int Max_cluster,vector<Rect>& totalRect,vector<Rect>& res,int width,int height){
    
    
    RNG rng(12345);
    int clusterCount=Max_cluster;
    int sampleCount=totalRect.size();
    cout<<"sampleCount is "<<sampleCount<<endl;
    Mat points(sampleCount,1,CV_32FC2),labels;
    clusterCount=MIN(clusterCount,sampleCount);
    
    Mat centers;//输出的聚类中心，kmeans的参数
    
    for(int nCenter=0;nCenter<sampleCount;nCenter++){
        Point center;
        center.x=totalRect[nCenter].x+int(totalRect[nCenter].width/2);
        center.y=totalRect[nCenter].y+int(totalRect[nCenter].height/2);
        points.at<Point2f>(nCenter)=center;
    }
    
    kmeans(points, clusterCount, labels,
           TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0),
           3, KMEANS_PP_CENTERS, centers);
    
    
    Point *p = new Point[clusterCount];
    
    cout<< "聚类中心"<< endl;
    
    for(int i = 0; i < clusterCount; ++i)
    {
        p[i] = (centers.ptr<Point2f>(i)[0]);
        cout<<"聚类"<<p[i]<< endl;
    }
    //找出属于该聚类中心的所有框
    
    for(int j=0;j<clusterCount;++j){
        vector<int> clusterIdx;
        for(int i = 0; i < sampleCount; i++ )
        {
            if(labels.at<int>(i)==j){
                clusterIdx.push_back(i);
            }
        }
        
        cout<<"j 聚类中心"<<j<<" 个数有 "<<clusterIdx.size()<<endl;
        // 求出聚类中心的最大覆盖区域，然后将rect的位置输出
        int xleft=10000;
        int xright=0;
        int yup=10000;
        int ydown=0;

        //        //输入的所有的矩形框rect  vect<Rect> totalRect;
        //        vector<Rect> totalRect;
        
        for(int mk=0;mk<clusterIdx.size();mk++){
            xleft=totalRect[clusterIdx[mk]].x<xleft?totalRect[clusterIdx[mk]].x:xleft;
            xleft=MAX(0,xleft);
            
            xright=(totalRect[clusterIdx[mk]].x+totalRect[clusterIdx[mk]].width)>xright?(totalRect[clusterIdx[mk]].x+totalRect[clusterIdx[mk]].width):xright;
            xright=MIN(width,xright);
            
            yup=totalRect[clusterIdx[mk]].y<yup?totalRect[clusterIdx[mk]].y:yup;
            yup=MAX(0,yup);
            ydown=(totalRect[clusterIdx[mk]].y+totalRect[clusterIdx[mk]].height)>ydown?(totalRect[clusterIdx[mk]].y+totalRect[clusterIdx[mk]].height):ydown;
            ydown=MIN(ydown,height);
            
        }
        
        // 测试完成以后，将当前的框全部保存输出
        Rect finalrect;
        finalrect.x=xleft;
        finalrect.width=xright-xleft;
        finalrect.y=yup;
        finalrect.height=ydown-yup;
        
        res.push_back(finalrect);
    }
    
}
