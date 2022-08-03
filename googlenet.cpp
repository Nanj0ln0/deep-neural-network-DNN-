#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace std;
using namespace cv::dnn;

//模型数据
String model_bin_file = "D:/OpenCV/Dnn model/Googlenet/bvlc_googlenet.caffemodel";
//描述文件(用于描述模型数据是什么结构)
String model_txt_file = "D:/OpenCV/Dnn model/Googlenet/bvlc_googlenet.prototxt";
//标签文件
String labels_txt_file = "D:/OpenCV/Dnn model/Googlenet/synset_words.txt";
vector<String>readLabels();

int main() {
	 
	Mat src = imread("D:/OpenCV/picture zone/cat.jpg");   //cat.jpg  spaceAirplane.jpg
	if (!src.data)
	{
		printf("ERROR");
		return -1;
	}
	namedWindow("input picture",CV_WINDOW_AUTOSIZE);
	
	//把标签文件读进来
	vector<String>labels = readLabels();
	//读模型
	Net net = readNetFromCaffe(model_txt_file,model_bin_file);
	if (net.empty())
	{
		printf("read caffe model data filure ...\n");
		return -2;
	}
	
	//输入图像，是否放缩，图像大小
	Mat inputBolb = blobFromImage(src , 1.0 , Size(224,224) , Scalar(104,117,123));
	Mat prob;
	for (int i = 0; i < 10; i++)
	{
		//输入，在网络的那一层
		net.setInput(inputBolb,"data");
		//返回可能性结果
		prob = net.forward("prob");
	}
	//将可能性图像  转换成一行多列的（1维的）
	Mat probMat = prob.reshape(1,1);

	Point classNumber;  //可能性最大地址
	double classProb;	//可能性最大值
	//输入可能性图像   寻找最大可能性
			//可能性图像 ，最小值 ，最大值，最小地址，最大地址
	minMaxLoc(probMat,NULL,&classProb,NULL,&classNumber);

	int classidx = classNumber.x;
	printf("\n current image classification:%s , possible:%.2f",labels.at(classidx).c_str(),classProb);
	//写字
	putText(src, labels.at(classidx),Point(20,20),FONT_HERSHEY_PLAIN,1.0,Scalar(0,0,255),2,8);
	
	 imshow("input picture", src);
	waitKey(0);
	return 0;

}

vector<String> readLabels()
{
	//分类名
	vector<String> className;
	//读取文件流
	ifstream fp(labels_txt_file);
	//未打开文件，退出
	if (!fp.is_open())
	{
		printf("counld not open\n");
		exit(-1);
	}

	string name;
	//判断文件是否读完
	while (!fp.eof())
	{
		//读取每一行，放在name中
		getline(fp,name); 
		//如果name中有长度则代表读到了
		if (name.length())
		{
			//从空格处分割  ，读取后面标签名
			className.push_back(name.substr(name.find(' ') + 1));
		}
	}
	fp.close();
	return className;
}
