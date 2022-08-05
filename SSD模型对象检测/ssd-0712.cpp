#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

//ѵ��ģ�����õ�300*300 ����ʶ��ͼ��ҲҪresize��300*300����ᱨ��
const size_t width = 300;
const size_t height = 300;

//ģ������
String model_bin_file = "D:/OpenCV/Dnn model/SSD0712Plus/VGG_VOC0712Plus_SSD_300x300_iter_240000.caffemodel";
//�����ļ�(��������ģ��������ʲô�ṹ)
String model_txt_file = "D:/OpenCV/Dnn model/SSD0712Plus/deploy.prototxt";
//��ǩ�ļ�
String labels_txt_file = "D:/OpenCV/Dnn model/SSD0712Plus/labelmap_det.txt";
vector<String>readLabels();

int main() {

	Mat src = imread("D:/OpenCV/picture zone/lena.jpg");  //lena.jpg  cat.jpg
	if (!src.data)
	{
		printf("ERROR");
		return -1;
	}
	namedWindow("input",CV_WINDOW_AUTOSIZE);
	imshow("input", src);


	//�ѱ�ǩ�ļ�������
	vector<String>objnames = readLabels();
	//����caffe SSD model
	Net net = readNetFromCaffe(model_txt_file, model_bin_file);

	//ͼ��Ԥ����
	Mat frame;
	src.copyTo(frame);
	if (frame.channels() == 4)
	{
		cvtColor(frame, frame, COLOR_BGRA2BGR);//��4ͨ��תΪ3ͨ��
	}

	if (frame.channels() == 1)
	{
		cvtColor(frame, frame, COLOR_GRAY2BGR);//����ͨ��תΪ3ͨ��

	}
	 Mat blobImage = blobFromImage(frame, 1.0f, Size(width, height), Scalar(104, 117, 123), false, false);

	//���룬���������һ�㿪ʼ  
		//.prototxt�ļ���ʼ
	net.setInput(blobImage,"data");
	//��������һ��
	Mat detection = net.forward("detection_out");
	/*
	net.forward���ص���һ��6ά��n��6�У�
	  ��0��		��1��	��2��	��3�� 		    ��4��					��5��
	center_x, center_y, width, height���þ��ο����һ����������Ŷ� ��Class1��������
	*/
		//���  �߶�  ��ʽ  ���ݲ��� 
	Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

	//���� ����������ֵ���������䣬�����޸ģ�Խ�ͼ�⵽������Խ��
	float confidence_threshold = 0.2;
	for (int i = 0; i < detectionMat.rows; i++)
	{
		float confidence = detectionMat.at<float>(i, 2);
		//�����򻭿��
		if (confidence > confidence_threshold)
		{
			size_t objIndex = (size_t)(detectionMat.at<float>(i, 1));
			//ʶ�����������    
			/*
				������ٴ洢��ʱ����һ����ԭͼ��С��һ������������Ҫ����ԭ���ĳ���
			*/
			//���Ͻ�
			float tl_x = detectionMat.at<float>(i, 3) * frame.cols;
			float tl_y = detectionMat.at<float>(i, 4) * frame.rows;
			//���½�
			float br_x = detectionMat.at<float>(i, 5) * frame.cols;
			float br_y = detectionMat.at<float>(i, 6) * frame.rows;

			//ǰ����4������
			Rect object_box((int)tl_x, (int)tl_y, (int)(br_x - tl_x), (int)(br_y - tl_y));
			rectangle(frame, object_box, Scalar(0, 0, 255), 2, 8, 0);
			putText(frame, format("%s", objnames[objIndex].c_str()), Point(tl_x, tl_y), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 0, 0), 2);

		}
	}

	imshow("ssd-demo", frame);
	
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
		getline(fp, name);
		//  ��ȡdisplay_name ��һ��
		if (name.length())
		{	
			/*
			//display_name:"��һ������14���ַ��Ǳ�ǩ��
			string temp = name.substr(14);
			//��Ϊ��ǩ�������һ�������ţ�Ҫȥ��
					//��Դ�ַ����ĵ�temp.end()-1���ַ���ʼ����������temp.end()��Ȼ����һ�����滻�ɿ�
			temp.replace(temp.end()-1 , temp.end() , "");
			*/
			
			string temp1 = name.substr(name.find(",")+1);
			string temp2 = temp1.substr(name.find(",") + 1);
			
			className.push_back(temp2);
		}
	}
	return className;
}