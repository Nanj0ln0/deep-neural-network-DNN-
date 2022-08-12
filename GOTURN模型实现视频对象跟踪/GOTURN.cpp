/*
在线下offline进行训练一个跟踪的模型，通过回归网络，用大量的数据进行训练。
训练完成后使用这个网络，对当前帧和前一帧进行对比。
先选取需要跟踪的区域，对区域进行卷积计算，对当前帧和前一帧进行对比，使用回归网络，来判断正确的位置。

这个模型只有跟踪的过程，没有训练的过程，所以速度很快，能达到理论上的帧
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace std;
using namespace cv::dnn;

String model = "D:/OpenCV/Dnn model/GOTURN/goturn.caffemodel";
String protetxt = "D:/OpenCV/Dnn model/GOTURN/goturn.prototxt";

Net net = readNetFromCaffe(protetxt, model);
Rect trackObjects(Mat &frame , Mat &prevframe);

Mat frame,prevFrame;
Rect prevBB;

int main() {

	VideoCapture capture;
	capture.open("D:/OpenCV/picture zone/car.mp4");
	if (!capture.isOpened())
	{
		printf("video is empty");
		return -1;
	}
	capture.read(frame);
	frame.copyTo(prevFrame);
	prevBB = selectROI(frame,true,true);
	namedWindow("frame",CV_WINDOW_AUTOSIZE);

	while (capture.read(frame))
	{
		Rect currentBB = trackObjects(frame, prevFrame);
		rectangle(frame, currentBB, Scalar(0, 0, 255), 2, 8, 0);

		// ready for next frame
		frame.copyTo(prevFrame);
		prevBB.x = currentBB.x;
		prevBB.y = currentBB.y;
		prevBB.width = currentBB.width;
		prevBB.height = currentBB.height;

		imshow("frame",frame);
		char c = waitKey(30);
		if (c == 27)
		{
			break;
		}
	}

	capture.release();

}


Rect trackObjects(Mat& frame, Mat& prevframe)
{
	Rect rect;
	//Using prevFrame & prevBB from model and curFrame GOTURN calculating curBB
	Mat curFrame = frame.clone();
	Rect2d curBB;

	float padTargetPatch = 2.0;
	Rect2f searchPatchRect, targetPatchRect;
	Point2f currCenter, prevCenter;
	Mat prevFramePadded, curFramePadded;
	Mat searchPatch, targetPatch;

	prevCenter.x = (float)(prevBB.x + prevBB.width / 2);
	prevCenter.y = (float)(prevBB.y + prevBB.height / 2);

	targetPatchRect.width = (float)(prevBB.width * padTargetPatch);
	targetPatchRect.height = (float)(prevBB.height * padTargetPatch);
	targetPatchRect.x = (float)(prevCenter.x - prevBB.width * padTargetPatch / 2.0 + targetPatchRect.width);
	targetPatchRect.y = (float)(prevCenter.y - prevBB.height * padTargetPatch / 2.0 + targetPatchRect.height);

	//为图像添加边框
	copyMakeBorder(prevFrame, prevFramePadded, (int)targetPatchRect.height, (int)targetPatchRect.height, (int)targetPatchRect.width, (int)targetPatchRect.width, BORDER_REPLICATE);
	targetPatch = prevFramePadded(targetPatchRect).clone();
	copyMakeBorder(curFrame, curFramePadded, (int)targetPatchRect.height, (int)targetPatchRect.height, (int)targetPatchRect.width, (int)targetPatchRect.width, BORDER_REPLICATE);
	searchPatch = curFramePadded(targetPatchRect).clone();

	//Preprocess
	//Resize
	resize(targetPatch, targetPatch, Size(227, 227));
	resize(searchPatch, searchPatch, Size(227, 227));

	//Mean Subtract
	targetPatch = targetPatch - 128;
	searchPatch = searchPatch - 128;

	//Convert to Float type
	targetPatch.convertTo(targetPatch, CV_32F);
	searchPatch.convertTo(searchPatch, CV_32F);

	Mat targetBlob = blobFromImage(targetPatch);
	Mat searchBlob = blobFromImage(searchPatch);

	net.setInput(targetBlob, "data1");
	net.setInput(searchBlob, "data2");

	Mat res = net.forward("scale");
	Mat resMat = res.reshape(1, 1);
	//printf("width : %d, height : %d\n", (resMat.at<float>(2) - resMat.at<float>(0)), (resMat.at<float>(3) - resMat.at<float>(1)));

	curBB.x = targetPatchRect.x + (resMat.at<float>(0) * targetPatchRect.width / INPUT_SIZE) - targetPatchRect.width;
	curBB.y = targetPatchRect.y + (resMat.at<float>(1) * targetPatchRect.height / INPUT_SIZE) - targetPatchRect.height;
	curBB.width = (resMat.at<float>(2) - resMat.at<float>(0)) * targetPatchRect.width / INPUT_SIZE;
	curBB.height = (resMat.at<float>(3) - resMat.at<float>(1)) * targetPatchRect.height / INPUT_SIZE;

	//Predicted BB
	Rect boundingBox = curBB;
	
	return boundingBox;
}
