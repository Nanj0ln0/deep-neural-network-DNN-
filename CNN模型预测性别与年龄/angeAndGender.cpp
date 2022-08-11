#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace std;
using namespace cv::dnn;

string age_model = "D:/OpenCV/Dnn model/cnnAgeAndGneder/age_net.caffemodel";
string age_prototxt = "D:/OpenCV/Dnn model/cnnAgeAndGneder/deploy_age.prototxt";
string gender_model = "D:/OpenCV/Dnn model/cnnAgeAndGneder/gender_net.caffemodel";
string gender_prototxt = "D:/OpenCV/Dnn model/cnnAgeAndGneder/deploy_gender.prototxt";

CascadeClassifier face_cascade;
String harrfile = "C:/Program Files/opencv/opencv-3.4.10/opencv/build/etc/haarcascades/haarcascade_frontalface_alt.xml";


void predict_age(Net &net ,Mat &image);
void predict_gender(Net &net ,Mat &image);

int main() {
	Mat src = imread("D:/OpenCV/picture zone/mans.jpg");
	if (!src.data)
	{
		printf("ERROR");
		return -1;
	}
	namedWindow("input",CV_WINDOW_AUTOSIZE);
	namedWindow("output",CV_WINDOW_AUTOSIZE);
	imshow("input", src);
	
	if (!face_cascade.load(harrfile))
	{
		printf("empty");
		return -2;
	}
	//级联分类器的人脸识别
	Mat src_gray;
	cvtColor(src, src_gray, COLOR_BGR2GRAY);
	equalizeHist(src_gray, src_gray);
	vector<Rect>face;
	face_cascade.detectMultiScale(src_gray, face, 1.1, 3, 0, Size(30, 30));


	Net ageNet = readNetFromCaffe(age_prototxt, age_model);
	Net genderNet = readNetFromCaffe(gender_prototxt, gender_model);


	for (size_t i = 0; i < face.size(); i++)
	{
		Mat roi = src(face[i]);
		rectangle(src, face[i], Scalar(30,30, 255), 2, 8, 0);
		predict_age(ageNet, roi);
		predict_gender(genderNet, roi);

	}

	imshow("output",src);
	waitKey(0);
	return 0;
}


//论文中的分类 = "https://talhassner.github.io/home/projects/cnn_agegender/CVPR2015_CNN_AgeGenderEstimation.pdf"
vector<String>agelable() {
	vector<String>ages;
	ages.push_back("0-2");
	ages.push_back("4-6");
	ages.push_back("8-13");
	ages.push_back("15-20");
	ages.push_back("25-32");
	ages.push_back("38-43");
	ages.push_back("48-53");
	ages.push_back("60-100");
	return ages;
}

void predict_age(Net& net, Mat& image)
{
	//输入
	Mat blobImage = blobFromImage(image, 1.0, Size(227, 227));
	net.setInput(blobImage,"data");
	//预测
	Mat age = net.forward("prob");
	Mat ageProb = age.reshape(1,1);//修改行数和通道数  变为一行单通道的
	Point classNum;
	double classProb;

	vector<String>ages = agelable();

	//筛选最大可能性
	minMaxLoc(ageProb,NULL,&classProb,NULL,&classNum);
	int classdix = classNum.x;
	//标记
	putText(image, format("ages:%s",ages.at(classdix).c_str()),Point(2,10),FONT_HERSHEY_PLAIN,0.8,Scalar(0,0,255),1);
}

void predict_gender(Net& net, Mat& image)
{
	//输入
	Mat blobImage = blobFromImage(image, 1.0, Size(227, 227));
	net.setInput(blobImage, "data");
	//预测	
	Mat gender = net.forward("prob");
	Mat genderProb = gender.reshape(1, 1);//修改行数和通道数  变为一行单通道的

	//标记
	putText(image, format("gender:%s", (genderProb.at<float>(0,0) > genderProb.at<float>(0,1)?"male":"famale")),
		Point(2, 20), FONT_HERSHEY_PLAIN, 0.8, Scalar(0, 0, 255), 1);

}
