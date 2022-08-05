#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

//训练模型是用的300*300 所以识别图像也要resize到300*300否则会报错
const size_t width = 300;
const size_t height = 300;

//模型数据
String model_bin_file = "D:/OpenCV/Dnn model/SSD0712Plus/VGG_VOC0712Plus_SSD_300x300_iter_240000.caffemodel";
//描述文件(用于描述模型数据是什么结构)
String model_txt_file = "D:/OpenCV/Dnn model/SSD0712Plus/deploy.prototxt";
//标签文件
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


	//把标签文件读进来
	vector<String>objnames = readLabels();
	//引入caffe SSD model
	Net net = readNetFromCaffe(model_txt_file, model_bin_file);

	//图像预处理
	Mat frame;
	src.copyTo(frame);
	if (frame.channels() == 4)
	{
		cvtColor(frame, frame, COLOR_BGRA2BGR);//将4通道转为3通道
	}

	if (frame.channels() == 1)
	{
		cvtColor(frame, frame, COLOR_GRAY2BGR);//将单通道转为3通道

	}
	 Mat blobImage = blobFromImage(frame, 1.0f, Size(width, height), Scalar(104, 117, 123), false, false);

	//输入，在网络的哪一层开始  
		//.prototxt文件开始
	net.setInput(blobImage,"data");
	//输出到最后一层
	Mat detection = net.forward("detection_out");
	/*
	net.forward返回的是一个6维（n行6列）
	  【0】		【1】	【2】	【3】 		    【4】					【5】
	center_x, center_y, width, height，该矩形框包含一个物体的置信度 ，Class1的自信心
	*/
		//宽度  高度  格式  数据部分 
	Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

	//设置 置信区间阈值，自信区间，可以修改，越低检测到的物体越多
	float confidence_threshold = 0.2;
	for (int i = 0; i < detectionMat.rows; i++)
	{
		float confidence = detectionMat.at<float>(i, 2);
		//大于则画框框
		if (confidence > confidence_threshold)
		{
			size_t objIndex = (size_t)(detectionMat.at<float>(i, 1));
			//识别对象的坐标点    
			/*
				坐标点再存储的时候是一个在原图大小的一个比例，所以要乘以原来的长宽
			*/
			//左上角
			float tl_x = detectionMat.at<float>(i, 3) * frame.cols;
			float tl_y = detectionMat.at<float>(i, 4) * frame.rows;
			//右下角
			float br_x = detectionMat.at<float>(i, 5) * frame.cols;
			float br_y = detectionMat.at<float>(i, 6) * frame.rows;

			//前景的4点坐标
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
		getline(fp, name);
		//  获取display_name 那一行
		if (name.length())
		{	
			/*
			//display_name:"那一行往后14个字符是标签名
			string temp = name.substr(14);
			//因为标签里面最后一个是引号，要去掉
					//从源字符串的第temp.end()-1个字符开始，往后数到temp.end()，然后将这一部分替换成空
			temp.replace(temp.end()-1 , temp.end() , "");
			*/
			
			string temp1 = name.substr(name.find(",")+1);
			string temp2 = temp1.substr(name.find(",") + 1);
			
			className.push_back(temp2);
		}
	}
	return className;
}