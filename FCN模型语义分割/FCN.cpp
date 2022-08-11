#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace std;
using namespace cv::dnn;

//模型数据
String model_bin_file = "D:/OpenCV/Dnn model/FCN/fcn8s-heavy-pascal.caffemodel";
//描述文件
String model_txt_file = "D:/OpenCV/Dnn model/FCN/deploy.prototxt";
//标签文件
String labels_txt_file = "D:/OpenCV/Dnn model/FCN/classes.txt";
vector<Vec3b>readColors();


int main() {
	Mat src = imread("D:/OpenCV/picture zone/bicycle match.jpg");
	if (!src.data)
	{
		printf("error");
		return -1;
	}
	namedWindow("input",CV_WINDOW_AUTOSIZE);
	namedWindow("output",CV_WINDOW_AUTOSIZE);
	imshow("input",src);
	//重新设置大小
	resize(src,src,Size(500,500));
	//读分类颜色
	vector<Vec3b> colors = readColors();
	//初始化加载网络
	Net net = readNetFromCaffe(model_txt_file,model_bin_file);
	Mat blobImage = blobFromImage(src);
	
	//使用网络
	net.setInput(blobImage,"data");
	Mat score = net.forward("score");

	//分割和优化
	const int channels = score.size[1];    //通道数
	const int rows = score.size[2];		//高
	const int cols = score.size[3];		//宽
	Mat maxColor(rows,cols,CV_8UC1);  //用于存储每一个颜色是什么东西的
	Mat maxValue(rows,cols,CV_32FC1);  //用于存储每一个东西是什么颜色	
	
	// 设置 LUT
	for (int c = 0; c < channels; c++)
	{
		for (int row = 0; row < rows; row++)
		{
			const float* ptrScore = score.ptr<float>(0,c,row);
			uchar* ptrMaxColor = maxColor.ptr<uchar>(row);
			float* ptrMaxValue = maxValue.ptr<float>(row);

			for (int col = 0; col < cols; col++)
			{
				if (ptrScore[col] > ptrMaxValue[col])
				{
					ptrMaxValue[col] = ptrScore[col];
					ptrMaxColor[col] = (uchar)c;
				}
			}
		}
	}

	//查找颜色
	Mat result = Mat::zeros(rows,cols,CV_8UC3);
	for (int row = 0; row < rows; row++)
	{
		const uchar* ptrMaxColor = maxColor.ptr<uchar>(row);
		Vec3b* ptrColor = result.ptr<Vec3b>(row);
		for (int col = 0; col < cols; col++)
		{
			ptrColor[col] = colors[ptrMaxColor[col]];
		}
	}

	Mat dst;
	addWeighted(src, 0.3, result, 0.7, 0, dst);
	imshow("output",dst);
		
	waitKey(0);
	return 0;
}

//读分类颜色
vector<Vec3b>readColors() {

	vector<Vec3b> colors;
	ifstream fp(labels_txt_file);
	if (!fp.is_open())
	{
		printf("counld not open\n");
		exit(-1);
	}

	string line;
	while (!fp.eof())
	{
		getline(fp, line);
		if (line.length())
		{
			stringstream ss(line);
			string name;
			ss >> name;

			int	temp;
			Vec3b color;
			ss >> temp;
			color[0] = (uchar)temp;
			ss >> temp;
			color[1] = (uchar)temp;
			ss >> temp;
			color[2] = (uchar)temp;
			colors.push_back(color);
	
		}
	}
	return colors;
}
