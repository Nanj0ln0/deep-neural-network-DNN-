#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace std;
using namespace cv::dnn;

//ģ������
String model_bin_file = "D:/OpenCV/Dnn model/Googlenet/bvlc_googlenet.caffemodel";
//�����ļ�(��������ģ��������ʲô�ṹ)
String model_txt_file = "D:/OpenCV/Dnn model/Googlenet/bvlc_googlenet.prototxt";
//��ǩ�ļ�
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
	
	//�ѱ�ǩ�ļ�������
	vector<String>labels = readLabels();
	//��ģ��
	Net net = readNetFromCaffe(model_txt_file,model_bin_file);
	if (net.empty())
	{
		printf("read caffe model data filure ...\n");
		return -2;
	}
	
	//����ͼ���Ƿ������ͼ���С
	Mat inputBolb = blobFromImage(src , 1.0 , Size(224,224) , Scalar(104,117,123));
	Mat prob;
	for (int i = 0; i < 10; i++)
	{
		//���룬���������һ��
		net.setInput(inputBolb,"data");
		//���ؿ����Խ��
		prob = net.forward("prob");
	}
	//��������ͼ��  ת����һ�ж��еģ�1ά�ģ�
	Mat probMat = prob.reshape(1,1);

	Point classNumber;  //����������ַ
	double classProb;	//���������ֵ
	//���������ͼ��   Ѱ����������
			//������ͼ�� ����Сֵ �����ֵ����С��ַ������ַ
	minMaxLoc(probMat,NULL,&classProb,NULL,&classNumber);

	int classidx = classNumber.x;
	printf("\n current image classification:%s , possible:%.2f",labels.at(classidx).c_str(),classProb);
	//д��
	putText(src, labels.at(classidx),Point(20,20),FONT_HERSHEY_PLAIN,1.0,Scalar(0,0,255),2,8);
	
	 imshow("input picture", src);
	waitKey(0);
	return 0;

}

vector<String> readLabels()
{
	//������
	vector<String> className;
	//��ȡ�ļ���
	ifstream fp(labels_txt_file);
	//δ���ļ����˳�
	if (!fp.is_open())
	{
		printf("counld not open\n");
		exit(-1);
	}

	string name;
	//�ж��ļ��Ƿ����
	while (!fp.eof())
	{
		//��ȡÿһ�У�����name��
		getline(fp,name); 
		//���name���г�������������
		if (name.length())
		{
			//�ӿո񴦷ָ�  ����ȡ�����ǩ��
			className.push_back(name.substr(name.find(' ') + 1));
		}
	}
	fp.close();
	return className;
}
