#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cmatch>

int main(int argc, char const *argv[])
{
	/* code */
	cv::Mat image1;
	cv::Mat image2(6, 6, CV_8UC1);
	cv::Mat image3(cv::Size(7, 7), CV_8UC3	);
	cv::Mat image4(8, 8, CV_32FC2, cv::Scalar(1,3));
	cv::Mat image5(cv::Size(9,9), CV_8UC3, cv::Scalar(1, 2, 3));
	cv::Mar image6(image2);

	std::cout<<image1<<std:: endl;


	cv::Mat Image1(10, 8, CV_8UC1, cv::Scalar(5));
	Image1.rows;
	Image1.cols;
	Image1.rowRange(1, 3);
	Image1.colRange(2, 4);

	cv::Mat Image2(8, 8, CV_32FC2, cv::Scalar(1,5));
	Image2.create(10, 10, CV_8UC3);
	Image2.channels();
	Image2.convertTo(Image2, CV_32F);
	Image2.depth();

	cv::Mat Image3 = cv::Mat::zeros(10, 10, CV_8UC1);
	Image1.row(4) = Image1.row(5)*2;
	Image1.col(4).copyTo(image1);


	cv::Mat srcImage = cv::imread("..\\images\\pool.jpg");
	if (srcImage.empty())
	{
		return -1;
	}
	cv::Mat srcGray;
	cv::cvtColor(srcImage, srcGray, CV_RGB2GRAY);
	cv::imshow("srcGray", srcGray);

	//均值平滑
	cv::Mat blurDstImage;
	blur(srcGray, blurDstImage, cv::Size(5, 5), cv::Point(-1, -1));
	cv::imshow("blurDstImage", blurDstImage);

	//写入文件
	cv::imwrite("blurDstImage.png", blurDstImage);
	cv::waitkey(0);

	cv::Mat resultImage(srcImage.size(), srcImage.type());
	cv::Mat xMapImage(srcImage.size(), CV_32FC1);
	cv::Mat yMapImage(srcImage.size(), CV_32FC1);
	int rows = srcImage.rows;
	int cols = srcImage.cols;
	for (int j = 0; j < rows; ++j)
	{
		for (int i = 0; i < cols; ++i)
		{
			xMapImage.at<float>(j, i) = cols - i;
			yMapImage.at<float>(j, i) = rows - j;
		}
	}
	remap(srcImage, resultImage, xMapImage, yMapImage, CV_INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));//重映射
	imshow("srcImage", srcImage);
	imshow("resultImage", resultImage);
	return 0;
}

//平移操作，大小不改变
cv::Mat imageTranslation1(cv::Mat &srcImage, int xOffset, int yOffset)
{
	int nRows = srcImage.rows;
	int nCols = srcImage.cols;
	cv::Mat resultImage(srcImage.size(), srcImage.type());
	for (int i = 0; i < nRows; ++i)
	{
		for (int j = 0; j < nCols; ++i)
		{
			int x = j - xOffset;
			int y = i - yOffset;
			//边界判断
			if (x >= 0 && y >= 0 && x < nCols && y < nRows)
			{
				resultImage.at<cv::Vec3b>(i, j) = srcImage.ptr<cv::Vec3b>(y)[x];
			}
		}
	}
	return resultImage;
}

//平移操作，大小改变
cv::Mat imageTranslation2(cv::Mat &srcImage, int xOffset, int yOffset)
{
	int nRows = srcImage.rows + abs(yOffset);
	int nCols = srcImage.cols + abs(xOffset);
	cv::Mat resultImage(nRows, nCols, srcImage.type());
	cv::Mat resultImage(srcImage.size(), srcImage.type());
	for (int i = 0; i < nRows; ++i)
	{
		for (int j = 0; j < nCols; ++i)
		{
			int x = j - xOffset;
			int y = i - yOffset;
			//边界判断
			if (x >= 0 && y >= 0 && x < nCols && y < nRows)
			{
				resultImage.at<cv::Vec3b>(i, j) = srcImage.ptr<cv::Vec3b>(y)[x];
			}
		}
	}
	return resultImage;
}

int main(int argc, char const *argv[])
{
	cv::Mat srcImage = cv::imread("..\\images\\lakewater.jpg");
	if (!srcImage.data)
	{
		return -1;
	}
	cv::imshow("srcImage", srcImage);
	int xOffset = 50, yOffset = 80;
	cv::Mat resultImage1 = imageTranslation1(srcImage, xOffset, yOffset);
	return 0;
}

using namespace cv;
//基于等间隔提取图形缩放
cv::Mat imageReduction1(cv::Mat $srcImage, float kx, float ky)
{
	int nRows = cvRound(srcImage.rows * kx);
	int nCols = cvRound(srcImage.cols * ky);
	cv::Mat resultImage(nRows, nCols, srcImage.type());
	for (int i = 0; i < nRows; ++i)
	{
		for (int j = 0; j < nCols; ++j)
		{
			int x = static_cast<int>((i+1)/kx + 0.5) - 1;
			int y = static_cast<int>((j+1)/ky + 0.5) - 1;
			resultImage.at<cv::Vec3b>(i, j) = srcImage.at<cv::Vec3b>(x, y);
		}
	}
	return resultImage;
}

cv::Vec3b areaAverage(const cv::Mat &srcImage, Point_<int> leftPoint, Point_<int> rightPoint)
{
	int temp1 = 0, temp2 = 0, temp3 = 0;
	int nPix = (rightPoint.x - leftPoint.x + 1)*(rightPoint.y - leftPoint.y + 1);
	for (int i = leftPoint.x; i < rightPoint.x; ++i)
	{
		for (int j = leftPoint.y; j < rightPoint.y; ++j)
		{
			temp1 += srcImage.at<cv::Vec3b>(i, j)[0];
			temp2 += srcImage.at<cv::Vec3b>(i, j)[1];
			temp3 += srcImage.at<cv::Vec3b>(i, j)[2];
		}
	}
	Vec3b vecTemp;
	vecTemp[0] = temp1 / nPix;
	vecTemp[1] = temp2 / nPix;
	vecTemp[2] = temp3 / nPix;
	return vecTemp;
}

//基于区域子块提取图形缩放
cv::Mat imageReduction2(const Mat &srcImage, double kx, double ky)
{
	int nRows = cvRound(srcImage.rows * kx);
	int nCols = cvRound(srcImage.cols * ky);
	cv::Mat resultImage(nRows, nCols, srcImage.type());
	int leftRowCoordinate = 0;
	int leftColCoordinate = 0;
	for (int i = 0; i < nRows; ++i)
	{
		int x = static_cast<int>((i+1)/kx + 0.5) - 1;
		for (int j = 0; i < nCols; ++j)
		{
			int y = static_cast<int>((j+1)/ky + 0.5) - 1;
			resultImage.at<Vec3b>(i, j) = areaAverage(srcImage, Point_<int>(leftRowCoordinate, rightRowCoordinate),Point_<int>(x, y));
			leftColCoordinate = y + 1;
		}
		leftColCoordinate = 0;
		leftRowCoordinate = x + 1;
	}
	return resultImage;
}

using namespace cv;
using namespace std;

cv::Mat angelRotate(Mat &src, int angle)
{
	float alpha = angle * CV_PI / 180;
	float rotateMat[3][3] = {
		{cos(alpha), -sin(alpha), 0},
		{sin(alpha), cos(alpha), 0},
		{0, 0, 1}
	};
	int nSrcRows = src.rows;
	int nSrcCols = src.cols;

	//计算旋转后各顶点位置
	float a1 = nSrcCols * rotateMat[0][0];
	float b1 = nSrcCols * rotateMat[1][0];
	float a2 = nSrcCols * rotateMat[0][0] + nSrcRows * rotateMat[0][1];
	float b2 = nSrcCols * rotateMat[1][0] + nSrcRows * rotateMat[1][1];
	float a3 = nSrcRows * rotateMat[0][1];
	float b3 = nSrcRows * rotateMat[1][1];

	float kxmin = min( min ( min(0.0f, a1), a2), a3);
	float kxmax = max( max ( max(0.0f, a1), a2), a3);
	float kymin = min( min ( min(0.0f, b1), b2), b3);
	float kymax = max( max ( max(0.0f, b1), b2), b3);

	int nRows = abs(kxmax - kxmin);
	int nCols = abs(kymax - kymin);
	Mat dst(nRows, nCols, src.type(), Scalar::all(0));
	for (int i = 0; i < nRows; ++i)
	{
		for (int j = 0; j < nCols; ++j)
		{
			int x = (j + kxmin) * rotateMat[0][0] - (i + kymin) * rotateMat[1][0];
			int y = -(j + kxmin) * rotateMat[1][0] + (i + kymin) * rotateMat[1][1];
			if (x >= 0 && y >= 0 && x < nSrcCols && y < nSrcRows)
			{
				dst.at<cv::Vec3b>(i, j) = src.at<cv::Vec3b>(y, x);
			}
		}
	}
	return dst;
}


int main(int argc, char const *argv[])
{
	Mat srcImage = imread("..\\images\\lakewater.jpg");
	if (!srcImage.data)
	{
		return -1;
	}
	imshow("srcImage", srcImage);
	Mat resultImage1 = imageReduction1(srcImage, 0.5, 0.5);
	imshow("resultImage1", resultImage1);
	Mat resultImage2 = imageReduction2(srcImage, 0.5, 0.5);
	imshow("resultImage2",resultImage2);
	int angle = 30;
	Mat resultImage = angelRotate(srcImage, angelRotate);
	imshow("resultImage", resultImage);
	//逆时针旋转90度
	Mat res1;
	transpose(srcImage,res1);
	
	Mar res2, res3, res4;
	flip(srcImage, res2, 1);//水平翻转
	flip(srcImage, res3, 0);//垂直翻转
	flip(srcImage, res4, -1);//垂直和水平翻转

	waitkey(0);

	return 0;
}

cv::Mat cv::getRotationMatrix2D(Point2f center, double angle, double scale)
{
	angle *= CV_PI / 180;
	double alpha = cos(angle) * scale;
	double beta = sin(angle) * scale;
	Mat M(2, 3, CV_64F);
	double* m = (double*)M.data;
	m[0] = alpha;
	m[1] = beta;
	m[2] = (1-alpha)*center.x - beta*center.y;
	m[3] = -beta;
	m[4] = alpha;
	m[5] = beta*center.x + (1-alpha)*center.y;
	return M;
}

int main(int argc, char const *argv[])
{
	Mat srcImage = imread("lena.jpg");
	if (!srcImage.data)
	{
		return -1;
	}
	imshow("srcImage", srcImage);
	int nRows = srcImage.rows;
	int nCols = srcImage.cols;
	Point2f srcPoint[3];
	Point2f resPoint[3];
	srcPoint[0] = Point2f(0, 0);
	srcPoint[1] = Point2f(nCols - 1, 0);
	srcPoint[2] = Point2f(0, nRows - 1);
	resPoint[0] = Point2f(nCols*0, nRows*0.33);
	resPoint[1] = Point2f(nCols*0.85, nRows*0.25);
	resPoint[2] = Point2f(nCols*0.15, nRows*0.7);
	Mat warpMat(Size(2, 3), CV_32F);
	Mat resultImage = Mat::zeros(nRows, nCols, srcImage.type());
	warpMat = getAffineTansform(srcPoint, resPoint);//点对点计算仿射变换矩阵
	warpAffine(srcImage, resultImage, warpMat, resultImage.size());
	imshow("resultImage", resultImage);

	//角度仿射变换
	Point2f centerPoint = Point2f(nCols/2, nRows/2);
	double angle = -50;
	double scale = 0.7;
	warpMat = getRotationMatrix2D(centerPoint, angle, scale);
	warpAffine(srcImage, resultImage, warpMat, resultImage.size());
	imshow("resultImage2", resultImage);

	return 0;
}

using namespace cv;
using namespace std;
int main(int argc, char const *argv[])
{
	VideoCapture capture;
	capture.open("..\\images\\testr.avi");
	if ( !capture.isOpened() )
	{
		cout<<"fail to open video!"<<endl;
		return -1;
	}
	long nTotalFrame = capture.get(CV_CAP_PROP_FAME_COUNT);
	cout<<"nTotalFrame = "<<nTotalFrame<<endl;

	int frameHeight = capture.get(CV_CAP_PROP_FAME_HEIGHT);
	int frameWidth = capture.get(CV_CAP_PROP_FAME_WIDTH);
	double framRate = capture.get(CV_CAP_PROP_FAME_FPS);

	Mat frameImg;
	long nCount = 1;
	while(true)
	{
		cout<<"current frame: "<<nCount<<endl;
		capture>>frameImg;
		if (!frameImg.empty())
		{
			imshow("frameImg", frameImg);
		}else{
			break;
		}
		if (char(waitkey(1)) == 'q')
		{
			break;
		}
		nCount++;
	}
	capture.release();

	VideoCapture capture(0);
	capture.set(CV_CAP_PROP_FAME_WIDTH, 400);
	capture.set(CV_CAP_PROP_FAME_HEIGHT, 400);
	return 0;
}

int main(int argc, char const *argv[])
{
	string sourceVideoPath = "..\\iamges\\test.avi";
	string outputVideoPath = "..\\images\\testWrite.avi";
	VideoCapture inputVideo(sourceVideoPath);
	if (!inputVideo.isOpened())
	{
		cout<<"fail to open!" <<endl;
		return -1;
	}
	VideoWrite outputVideo;
	cv::Size videoResolution = cv::Size((int)inputVideo.get(CV_CAP_PROP_FAME_WIDTH), (int)inputVideo.get(CV_CAP_PROP_FAME_HEIGHT));
	double fps = inputVideo.get(CV_CAP_PROP_FAME_FPS);
	outputVideo.open(outputVideoPath, -1, 25.0, videoResolution, true);
	if (!outputVideo.isOpened())
	{
		cout<< "fail to open!" <<endl;
	}
	cv::Mat frameImg;
	int count = 0;
	std::vector<cv::Mat> rgb;
	cv::Mat resultImg;
	while(true)
	{
		inputVideo >> frameImg;
		if (!frameImg.empty())
		{
			count++;
			imshow("frameImg", frameImg);
			split(frameImg, rgb);
			for (int i = 0; i < 3; ++i)
			{
				if (i != 0)
				{
					rgb[i] = Mat::zeros(videoResolution, rgb[0].type());
				}
				merge(rgb, resultImg);
			}
			imshow("resultImg", resultImg);
			outputVideo << resultImg;
		}else{
			break;
		}
	}
	return 0;
}

Mat srcImage;
void MouseEvent(int event, int x, int y, int flags, void* data)
{
	char charText[30];
	Mat tempImage, hsvImage;
	tempImage = srcImage.clone();
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		Vec3b p = tempImage.at<Vec3b>(y, x);
		sprintf(charText, "R=%d, G=%d, B=%d", p[2], p[1],p[0]);
		putText(tempImage, charText, Point(8, 20), FONT_HERSHEY_PLAIN, 2, CV_RGB(255, 0, 0));
		imwrite("..\\images\\HSVFlower4.jpg", tempImage);
	}else{
		sprintf(charText, "x=%d, y=%d", x, y);
		putText(tempImage, charText, Point(8, 20), FONT_HERSHEY_PLAIN, 2, CV_RGB(255, 0, 0));
		imwrite("..\\images\\NOFlower4.jpg", tempImage);
	}
	imshow("srcImage", tempImage);
}

void onChangeTrackBar(int pos, void* data)
{
	Mat srcImage = *(Mat*)(data);
	Mat dstImage;
	threshold(srcImage, dstImage, pos, 255, 0);
	imshow("dyn_threshold", dstImage);
}

int main(int argc, char const *argv[])
{
	Mat srcImage = imread("..\\images\\flower.png");
	if ( !srcImage.data)
	{
		return -1;
	}
	Mat srcGray;
	cvtColor(srcImage, srcGray, CV_RGB2GRAY);
	namedWindow("dyn_threshold");
	imshow("dyn_threshold",srcGray);
	createTrackbar("pos", "dyn_threshold", 0, 255, onChangeTrackBar, &srcGray);
	waitkey(0);
	return 0;
}
int main(int argc, char const *argv[])
{
	srcImage = imread("..\\images\\flower.jpg");
	if (srcImage.empty())
	{
		return -1;
	}
	namedWindow("srcImage");
	setMouseCallback("srcImage", MouseEvent, 0);
	imshow("srcImage",srcImage);
	waitkey(0);
	return 0;
}

Mat srcImage;
void reigonExtraction(int xRoi, int yRoi, int widthRoi, int heightRoi){
	Mat roiImage(srcImage.rows, srcImage.cols, CV_8UC3);
	srcImage(Rect(xRoi, yRoi, widthRoi, heightRoi)).copyTo(roiImage);
	imshow("roiImage",roiImage);
	waitkey(0);
}
int main(int argc, char const *argv[])
{
	srcImage = imread("..\\images\\flower.png");
	if (!srcImage.data)
	{
		return 1;
	}
	imshow("srcImage",srcImage);
	waitkey(0);
	int xRoi = 80;
	int yRoi = 180;
	int widthRoi = 150;
	int heightRoi = 100;
	reigonExtraction(xRoi, yRoi, widthRoi, heightRoi);
	return 0;
}

Mat srcImage;
Rect roiRect;
Point startPoint;
Point endPoint;
bool downFlag = false;
bool upFlag = false;
void MouseEvent(int event, int x, int y, int flags, void* data)
{
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		downFlag = true;
		startPoint.x = x;
		startPoint.y = y;
	}
	if (event == CV_EVENT_LBUTTONUP)
	{
		upFlag = true;
		endPoint.x = x;
		endPoint.y = y;
	}
	if (downFlag == true && upFlag == false)
	{
		Point tempPoint;
		tempPoint.x = x;
		tempPoint.y = y;
		Mat tempImage = srcImage.clone();
		rectangle(tempImage,startPoint,tempPoint,Scalar(255, 0, 0), 2, 3, 0);
		imshow("ROIing", tempImage);
	}
	if (downFlag == true && upFlag == true)
	{
		roiRect.width = abs(startPoint.x - endPoint.x);
		roiRect.height = abs(startPoint.y - endPoint.y);
		roiRect.x = min(startPoint.x, endPoint.x);
		roiRect.y = min(startPoint.y, endPoint.y);
		Mat roiMat(srcImage, roiRect);
		imshow("ROI", roiMat);
		downFlag = false;
		upFlag = false;
	}
	waitkey(0);
}

//图像元素遍历--反色
Mat inversrColor1(Mat srcImage)
{
	Mat tempImage = srcImage.clone();
	int row = tempImage.rows;
	int col = tempImage.cols;
	for (int i = 0; i < row; ++i)
	{
		for (int j = 0; j < col; ++j)
		{
			tempImage.at<Vec3b>(i, j)[0] = 255 - tempImage.at<Vec3b>(i, j)[0];
			tempImage.at<Vec3b>(i, j)[1] = 255 - tempImage.at<Vec3b>(i, j)[1];
			tempImage.at<Vec3b>(i, j)[2] = 255 - tempImage.at<Vec3b>(i, j)[2];
		}
	}
	return tempImage;
}

Mat inversrColor2(Mat srcImage)
{
	Mat tempImage = srcImage.clone();
	int row = tempImage.rows;
	int nstep = tempImage.cols * tempImage.channels();
	for (int i = 0; i < row; ++i)
	{	
		const uchar* pSrcData = srcImage.ptr<uchar>(i);
		uchar* pResData = tempImage.ptr<uchar>(i);
		for (int j = 0; j < nstep; ++j)
		{
			pResData[j] = saturate_cast<uchar>(255 - pSrcData[j]);
		}
	}
	return tempImage;
}

using namespace cv;
void showManyImages(const std::vector<Mat> &srcImages, cv::Size imgSize)
{
	int nNumImages = srcImages.size();
	cv::Size nSizeWindows;
	if (nNumImages > 12)
	{
		cout << "Not more than 12 mages" <<endl;
		return;
	}
	switch(nNumImages)
	{
		case 1: nSizeWindows = cv::Size(1, 1);break;
		case 2: nSizeWindows = cv::Size(2, 1);break;
		case 3:
		case 4: nSizeWindows = cv::Size(2, 2);break;
		case 5: 
		case 6: nSizeWindows = cv::Size(3, 2);break;
		case 7: 
		case 8: nSizeWindows = cv::Size(4, 2);break;
		case 9: nSizeWindows = cv::Size(3, 3);break;
		default: nSizeWindows = cv::Size(4, 3);break;
	}
	int nShowImageSize = 200;
	int nSplitLineSize = 15;
	int nAroundLineSize = 50;
	const int imagesHeight = nShowImageSize * nSizeWindows.width + nAroundLineSize +(nSizeWindows.width - 1)* nSplitLineSize;
	const int imagesWidth = nShowImageSize * nSizeWindows.height + nAroundLineSize +(nSizeWindows.height - 1)* nSplitLineSize;
	Mat showWindowImages(imagesWidth, imagesHeight, CV_8UC3, Scalar(0, 0, 0));
	int posX = (showManyImages.cols - (nShowImageSize*nSizeWindows.width + (nSizeWindows.width - 1)* nSplitLineSize))/2;
	int posY = (showManyImages.rows - (nShowImageSize*nSizeWindows.height + (nSizeWindows.height - 1)* nSplitLineSize))/2;
	int tempPosX = posX;
	int tempPosY = posY;
	for (int i = 0; i < nNumImages; ++i)
	{
		if ((i%nSizeWindows.width == 0) && (tempPosX != posX))
		{
			tempPosX = posX;
			tempPosY = += (nSplitLineSize + nShowImageSize);
		}
		Mat tempImage = showWindowImages(Rect(tempPosX, tempPosY, nShowImageSize,nShowImageSize));
		resize(srcImage[i], tempImage, Size(nShowImageSize, nShowImageSize));
		tempPosX += (nSplitLineSize + nShowImageSize);
	}
	imshow("showWindowImages", showWindowImages);

}

int main(int argc, char const *argv[])
{
	const int num = 100;
	char fileName[50];
	char windowName[50];
	Mat srcImage;
	for (int i = 0; i <= num; ++i)
	{
		sprintf(fileName, "D:\\test\\1(%d).jpg", i);
		sprintf(windowName, "NO%d", i);
		srcImage = imread(fileName);
		namedWindow(windowName);
		imshow(windowName,srcImage);
		waitkey(0);
	}
	return 0;
}

//OTSU二值化
#include <stdio.h>
#include <string>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;
int OTSU(Mat srcImage)
{
	int nCols = srcImage.cols;
	int nRows = srcImage.rows;
	int threshold = 0;
	int nSumPix[256];
	float nProDis[256];
	for (int i = 0; i < 256; ++i)
	{
		nSumPix[i] = 0;
		nProDis[i] = 0;
	}
	for (int i = 0; i < nCols; ++i)
	{
		for (int j = 0; j < nRows; ++j)
		{
			nSumPix[(int)srcImage.at<uchar>(i, j)]++;
		}
	}
	for (int i = 0; i < 256; ++i)
	{
		nProDis[i] = (float)nSumPix[i] / (nCols * nRows);
	}
	float w0, w1, u0_temp, u1_temp, u0, u1, delta_temp;
	double delta_max = 0.0;
	for (int i = 0; i < 256; ++i)
	{
		w0 = w1 = u0_temp = u1_temp = u0 = u1 = delta_temp = 0;
		for (int j = 0; j < 256; ++j)
		{
			if (j <= i)//背景部分
			{
				w0 += nProDis[j];
				u0_temp += j * nProDis[j];
			}else{
				w1 += nProDis[j];
				u1_temp += j * nProDis[j];
			}
		}
		u0 = u0_temp / w0;
		u1 = u1_temp / w1;
		delta_temp = (float)(w0 *w1 * pow((u0-u1), 2));
		if (delta_temp > delta_max)
		{
			delta_max = delta_temp;
			threshold = i;
		}
	}
	return threshold;
}
int main(int argc, char const *argv[])
{
	srcImage = imread("..\\images\\flower.png");
	if (!srcImage.data)
	{
		return 1;
	}
	Mat srcGray;
	cvtColor(srcImage, srcGray, CV_RGB2GRAY);
	imshow("srcGray", srcGray);
	int ostuThreshold = OTSU(srcGray);
	cout << ostuThreshold << endl;
	Mat otsuResultImage = Mat::zeros(srcGray.rows, srcGray.cols, CV_8UC1);
	for (int i = 0; i < srcGray.rows; ++i)
	{
		for (int j = 0; j < srcGray.cols; ++j)
		{
			if (srcGray.at<uchar>(i, j) > ostuThreshold)
			{
				srcGray.at<uchar>(i, j) = 255;
			}else{
				srcGray.at<uchar>(i, j) = 0;
			}
		}
	}
	imshow("otsuResultImage", otsuResultImage);

	Mat dstImage;
	int thresh = 130;
	int threshType = 0;
	const int maxVal = 255;
	threshold(srcGray, dstImage, thresh, maxVal, threshType);//固定阀值
	adaptiveThreshold(srcGray, dstImage, maxVal, adaptiveMethod, threshType, blockSize, constValue);//自适应阀值

	//双阀值化
	threshold(srcGray, dstImage1, low_thresh, maxVal, THRESH_BINARY);
	threshold(srcGray, dstImage2, high_thresh, maxVal, THRESH_BINARY_INV);
	bitwise_and(dstImage1, dstImage2, dstImage);

	//半阀值化
	threshold(srcGray, dstImage1, low_thresh, maxVal, THRESH_BINARY);
	bitwise_and(srcGray, dstImage1, dstImage);
	
	imshow("dstImage", dstImage);
	waitkey(0);
	return 0;
}

#include <opencv2\opencv.hpp>
using namespace cv;
int main(int argc, char const *argv[])
{
	Mat Image, ImageGray, hsvMat;
	Image = imread("..\\iamges\\flower.png");
	if (Image.empty())
	{
		return -1;
	}
	imshow("Image", Image);
	cvtColor(Image, ImageGray, CV_BGR2GRAY);
	const int channels[1] = {0};
	const int histSize[1] = {256};
	float pranges[2] = {0, 255};
	const float* ranges[1] = {pranges};
	MatND hist;
	calcHist(&ImageGray, 1, channels, Mat(), hist, 1, histSize, ranges);

	int hist_w = 500;
	int hist_h = 500;
	int nHistSize = 255;
	int bin_w = cvRound((double) hist_w/nHistSize);
	Mat histImage(hist_w, hist_h, Scalar(0, 0, 0));
	nomalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	for (int i = 0; i < nHistSize; ++i)
	{
		line(histImage, Point( bin_w*(i-1), hist_h-cvRound(hist.at<float>(i-1))), Point( bin_w*(i), hist_h-cvRound(hist.at<float>(i))), Scalar(0, 0, 255), 2, 8, 0);
	}
	imshow("histImage", histImage);
	waitkey(0);

	cvtColor(Image, hsvMat, CV_BGR2HSV);
	int hbins = 30, sbins = 32;
	int histSize[] = {hbins, sbins};
	float hranges =  {0, 180};
	float sranges = {0, 256};
	const float* ranges[] = {hranges, sranges};
	calcHist(&hsvMat, 1, channels, Mat(), hist, 2, histSize, ranges, true, false);
	double maxVal = 0;
	minMaxLoc(hist, 0, &maxVal, 0, 0);
	int scale = 10;
	Mat histImg = Mat::zeros(sbins*scale, hbins*10, CV_8UC3);
	for (int h = 0; h < hbins; ++h)
	{
		for (int s = 0; s < sbins; ++s)
		{
			float binVal = hist.at<float>(h, s);
			int intensity = cvRound(binVal*255/maxVal);
			rectangle(histImg, Point(h*scale, s*scale),Point((h+1)*scale-1, (s+1)*scale - 1), Scalar::all(intensity), CV_FILLED);
		}
	}
	imshow("Source", Image);
	imshow("H-S Histogram", histImg);
	waitkey(0);

	std::vector<Mat> bgr_planes;
	split(srcImages, bgr_planes);
	int histSize = 256;
	float range[] = {0, 256};
	const float* histRange = {range};
	bool uniform = true;
	bool accumulate = false;
	Mat b_hist, g_hist, r_hist;
	calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate); 
	return 0;
}

int main(int argc, char const *argv[])
{
	Mat Image, ImageGray;
	Image = imread("..\\iamges\\flower.png");
	if (Image.empty())
	{
		return -1;
	}
	imshow("Image", Image);
	cvtColor(Image, ImageGray, CV_BGR2GRAY);
	Mat heqResult;
	equalizeHist(ImageGray, heqResult);
	imshow("heqResult", heqResult);

	Mat colorHeqImage;
	std::vector<Mat> BGR_planes;
	split(Image, BGR_planes);
	for (int i = 0; i < BGR_planes.size(); ++i)
	{
		equalizeHist(BGR_planes[i], BGR_planes[i]);

	}
	merge(BGR_planes, colorHeqImage);
	imshow("colorHeqImage", colorHeqImage);


	Mat srcImage = imread("..\\images\\flower.png");
	if (!srcImage.data )
	{
		return 1;
	}
	Mat srcGray;
	cvtColor(srcImage, srcGray, CV_BGR2GRAY);
	const int channels[1] = {0};
	const int histSize[1] = {256};
	float hranges[2] = {0, 255};
	const float* ranges[1] = {hranges};
	MatND hist;
	calcHist(&srcGray, 1, channels, Mat(), hist, 1, histSize,ranges);
	int segThreshold = 50;
	int iLow = 0;
	for (; iLow < histSize[0]; ++iLow)
	{
		if (hist.at<float>(iLow) > segThreshold)
		{
			break;
		}
	}
	int iHigh = histSize[0] - 1;;
	for (; iHigh >= 0; iHigh--)
	{
		if (hist.at<float>(iHigh) > segThreshold)
		{
			break;
		}
	}
	Mat lookUpTable(Size(1, 256), CV_8U);
	for (int i = 0; i < 256; ++i)
	{
		if (i<iLow)
		{
			lookUpTable.at<uchar>(i) = 0;
		}else if (i > iHigh)
		{
			lookUpTable.at<uchar>(i) = 255;
		}else{
			lookUpTable.at<uchar>(i) = static_cast<uchar>( 255.0*(i - iLow)/(iHigh - iLow + 0.5) );
		}
	}
	Mat histTransResult;
	LUT(srcGray, lookUpTable, histTransResult);
	imshow("histTransResult", histTransResult);

	//直方图变换累计
	float table[256];
	int nPix = srcGray.cols*srcGray.rows;
	for (int i = 0; i < 256; ++i)
	{
		float temp[256];
		temp[i] = hist.at<float>(i)/nPix*255;
		if (i!=0)
		{
			table[i] = table[i-1] + temp[i];
		}else{
			table[i] = temp[i];
		}
	}
	Mat lookUpTable(Size(1, 256), CV_8U);
	for (int i = 0; i < 256; ++i)
	{
		lookUpTable.at<uchar>(i) = static_cast<uchar>( table[i] );
	}
	Mat histTransResult;
	LUT(srcGray, lookUpTable, histTransResult);
	imshow("histTransResult", histTransResult);
	waitkey(0);
	return 0;
}

using namespace cv;
using namespace std;
int main(int argc, char const *argv[])
{
	Mat srcImage = imread("..\\images\\flower.png");
	Mat dstImage = imread("..\\images\\sea.jpg");
	if ( !srcImage.data || !dstImage.data)
	{
		return 1;
	}
	resize(dstImage, dstImage, Size(srcImage.rows, srcImage.cols), 0, 0, CV_INTER_LINEAR);
	imshow("srcImage", srcImage);
	imshow("dstImage", dstImage);
	waitkey(0);
	//初始化累计分布参数
	float srcCdfArr[256];
	float dstCdfArr[256];
	int srcAddTemp[256];
	int dstAddTemp[256];
	int histMatchTemp[256];
	for (int i = 0; i < 256; ++i)
	{
		srcAddTemp[i] = 0;
		dstAddTemp[i] = 0;
		srcCdfArr[i] = 0;
		dstCdfArr[i] = 0;
		histMatchMap[i] = 0;
	}
	float sumSrcTemp = 0;
	float sumDstTemp = 0;
	int nSrcPix = srcImage.cols * srcImage.rows;
	int nDstPix = dstImage.clos * dstImage.rows;
	int matchFlag = 0;
	//求解源图像与目标图像的累计直方图
	for (size_t nrow = 0; nrow < srcImage.rows; ++nrow)
	{
		for (size_t ncol = 0; ncol < srcImage.cols; ++ncol)
		{
			srcAddTemp[(int)srcImage.at<uchar>(nrow, ncol)]++;
			dstAddTemp[(int)srcImage.at<uchar>(nrow, ncol)]++;
		}
	}
	//求解源图像与目标图像的累计概率分布
	for (int i = 0; i < 256; ++i)
	{
		sumSrcTemp += srcAddTemp[i];
		srcCdfArr[i] = sumSrcTemp / nSrcPix;
		sumDstTemp += dstAddTemp[i];
		dstCdfArr[i] = sumDstTemp / nDstPix;
	}
	//直方图匹配实现
	for (int i = 0; i < 256; ++i)
	{
		float minMatchPara = 20;
		for (int j = 0; j < 256; ++j)
		{
			if (minMatchPara > abs(srcCdfArr[i] - dstCdfArr[j]))
			{
				minMatchPara = abs(srcCdfArr[i] - dstCdfArr[j]);
				matchFlag = j;
			}
		}
		histMatchMap[i] = matchFlag;
	}
	Mat HistMatchImage = Mat::zeros(srcImage.rows, srcImage.cols, CV_8UC3);
	cvtColor(srcImage, HistMatchImage, CV_BGR2GRAY);
	for (int i = 0; i < HistMatchImage.rows; ++i)
	{
		for (int j = 0; j < HistMatchImage.cols; ++j)
		{
			HistMatchImage.at<uchar>(i, j) = histMatchMap[(int)HistMatchImage.at<uchar>(i, j)];
		}
	}
	imshow("resultImage", HistMatchImage);
	return 0;
}

//图像灰度变换技术：
//1,直方图处理技术
//2，距离变换
//3，Gamma 校正
//4.线性变换、对数变换、分段线性

Mat gammaTransform(Mat& srcImage, float kFactor)//gamma校正
{
	unsigned char LUT[256];
	for (int i = 0; i < 256; ++i)
	{
		LUT[i] = saturate_cast<uchar>(pow((float)(i/255.0), kFactor) * 255.0f);
	}
	Mat resultImage = srcImage.clone();
	if (srcImage.channels() == 1)//单通道
	{
		MatIterator_<uchar> iterator = resultImage.begin<uchar>();
		MatIterator_<uchar> iteratorEnd = resultImage.end<uchar>();
		for (; iterator != iteratorEnd; iterator++)
		{
			*iterator = LUT[(*iterator)];
		}
	}else{//3通道
		MatIterator_<Vec3b> iterator = resultImage.begin<Vec3b>();
		MatIterator_<Vec3b> iteratorEnd = resultImage.end<Vec3b>();
		for (; iterator != iteratorEnd; iterator++)
		{
			(*iterator)[0] = LUT[((*iterator)[0])];
			(*iterator)[1] = LUT[((*iterator)[1])];
			(*iterator)[2] = LUT[((*iterator)[2])];
		}
	}
	return resultImage;
}

//最大熵阀值分割
using namespace cv;
using namespace std;

//计算当前位置的能量熵
float caculateCurrentEntropy(Mat hist, int threshold)
{
	float backgroundSum = 0, targetSum = 0;
	const float* pDataHist = (float*) hist.ptr<float>(0);
	for (int i = 0; i < 256; ++i)
	{
		if (i < threshold)//累计背景值
		{
			backgroundSum += pDataHist[i];
		}else//累计目标值
		{
			targetSum += pDataHist[i];
		}
	}
	float backgroundEntropy = 0, targetEntropy = 0;
	for (int i = 0; i < 256; ++i)
	{
		if (i < threshold)//背景熵
		{
			if (pDataHist[i] == 0) continue;
			float ratio1 = pDataHist[i] / backgroundSum;
			backgroundEntropy += -ratio1 * logf(ratio1);
		}else//目标熵
		{
			if (pDataHist[i] == 0) continue;
			float ratio2 = pDataHist[i] / backgroundSum;
			backgroundEntropy += -ratio2 * logf(ratio2);
		}
	}
	return targetEntropy + backgroundEntropy;
}

//寻找最大熵阀值并分割
Mat maxEntropySegMentation(Mat inputImage)
{
	const int channels[1] = {0};
	const int histSize[256] = {256};
	float pranges[2] = {0, 256};
	const float* ranges[1] = {pranges};
	MatND hist;
	calcHist(&inputImage, 1, channels, Mat(), hist, 1, histSize, ranges);//直方图
	float maxentropy = 0;
	int   max_index = 0;
	Mat result;
	for (int i = 0; i < 256; ++i)
	{
		float cur_entropy = caculateCurrentEntropy(hist, i);
		if (cur_entropy > maxentropy)
		{
			maxentropy = cur_entropy;
			max_index = i;
		}
	}
	threshold(inputImage, result, max_index, 255, CV_THRESH_BINARY);
	return result;
}
int main(int argc, char const *argv[])
{
	Mat srcImage, grayImage;
	srcImage = imread("..\\iamges\\flower.png");
	if (!srcImage.data)
	{
		return 0;
	}
	cvtColor(srcImage, grayImage, CV_BGR2GRAY);
	Mat result = maxEntropySegMentation(grayImage);
	imshow("grayImage", grayImage);
	imshow("result", result);
	return 0;
}

#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

//导向滤波
Mat guidedfilter(Mat &srcImage, Mat &srcClone, int r, double eps)
{
	srcImage.convertTo(srcImage, CV_64FC1);
	srcClone.convertTo(srcClone, CV_64FC1);
	int nRows = srcImage.rows;
	int nCols = srcImage.cols;
	//计算均值
	Mat boxResult;
	boxFilter(Mat::ones(nRows, nCols, srcImage.type()), boxResult, CV_64FC1, Size(r, r));
	//生产导向均值
	Mat mean_I;
	boxFilter(srcImage, mean_I, CV_64FC1, Size(r, r));
	//生产原始均值
	Mat mean_P;
	boxFilter(srcImage, mean_P, CV_64FC1, Size(r, r));
	//生产互相关均值
	Mat mean_IP;
	boxFilter(srcImage.mul(srcClone), mean_IP, CV_64FC1, Size(r, r));
	Mat cov_IP = mean_IP - mean_I.mul(mean_P);
	//生产自相关均值
	Mat mean_II;
	boxFilter(srcImage.mul(srcImage), mean_II, CV_64FC1, Size(r, r));
	Mat var_I  = mean_II - mean_I.mul(mean_I);
	Mat var_IP  = mean_Ip - mean_I.mul(mean_P);
	Mat a = cov_IP/(var_I + eps);
	Mat b = mean_P - a.mul(mean_I);
	Mat mean_a;
	boxFilter(a, mean_a, CV_64FC1, Size(r, r));
	mean_a = mean_a / boxResult;
	Mat mean_b;
	boxFilter(b, mean_b, CV_64FC1, Size(r, r));
	mean_b = mean_b / boxResult;
	resultMat = mean_a.mul(srcImage) + mean_b;
	return resultMat;
}

int main(int argc, char const *argv[])
{
	Mat srcImage = imread("..\\images\\flower.png");
	if (srcImage.empty())
	{
		return -1;
	}
	std::vector<Mat> vSrcImage, vResultImage;
	split(srcImage, vSrcImage);
	Mat resultMat;
	for (int i = 0; i < 3; ++i)
	{
		Mat tempImage;
		vSrcImage[i].convertTo(tempImage, CV_64FC1, 1.0/255.0);
		Mat p = tempImage.clone();
		Mat resultImage = guidedfilter(tempImage, p, 4, 0.01);
		vResultImage.push_back(resultImage);
	}
	merge(vResultImage, resultMat);
	imshow("srcImage", srcImage);
	imshow("resultMat", resultMat);
	waitkey(0);
	return 0;
}

//图像污点修复
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/photo/photo.hpp"
#include <opencv2/opencv.cpp>
#include <iostream>

using namespace std;
using namespace cv;

Mat img, inpaintMask;
Point prevPt(-1, -1);
static viod onMouse(int event, int x, int y, int flags, void*)
{
	if (event == CV_EVENT_LBUTTONUP || !(flags & CV_EVENT_FLAG_LBUTTON))
	{
		prevPt = Point(-1, -1);
	}else if (event == CV_EVENT_LBUTTONDOWN)
	{
		prevPt = Point(x, y);
	}else if (event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_LBUTTONU))
	{
		Point pt(x, y);
		if (prevPt.x < 0) prevPt = pt;
		line(inpaintMask, prevPt, pt, Scalar::all(255), 5, 8, 0);
		line(img, prevPt, pt, Scalar::all(255), 5, 8, 0);
		prevPt = pt;
		imshow("image", img);
	}
}

int main(int argc, char const *argv[])
{
	Mat src = imread("..\\images\\flower.png");
	if (src.empty())
	{
		return 0;
	}
	img = src.clone();
	inpaintMask = Mat::zeros(img.size(),CV_8U);
	imshow("srcImage", img);
	setMouseCallback("srcImage", onMouse, 0);
	for (; ; )
	{
		char c = (char)waitkey();
		if (c == 27)
		{
			break;
		}
		if (c == "r")
		{
			Mat resMat;
			inpaintMask(img, inpaintMask, resMat, 3, CV_INPAINT_TELEA);
			imshow("resMat", resMat);
		}
	}
	return 0;
}
//旋转文本图像矫正
int main(int argc, char const *argv[])
{
	Mat srcImage = imread("..\\images\\flower.png");
	if (src.empty())
	{
		return 0;
	}
	cvtColor(srcImage, srcImage, CV_RGB2GRAY);
	const int nRows = srcImage.rows;
	const int nCols = srcImage.cols;
	//获取DFT尺寸
	int cRows = getOptimalDFTSize(nRows);
	int cCols = getOptimalDFTSize(nCols);
	Mat sizeConvMat;
	copyMakeBorder(srcImage, sizeConvMat, 0, cRows - nRows, 0, cCols - nCols, BORDER_CONSTANT, Scalar::all(0) );
	imshow("sizeConvMat", sizeConvMat);
	//图像DFT变换
	Mat groupMats[] = {Mat_<float>(sizeConvMat), Mat::zeros(sizeConvMat.size(), CV_32F)};
	Mat mergeMat;
	merge(groupMats, 2, mergeMat);
	dft(mergeMat, mergeMat);
	split(mergeMat, groupMats);
	magnitude(groupMats[0], groupMats[1], groupMats[0]);
	Mat magnitudeMat = groupMats[0].clone();
	magnitudeMat += Scalar::all(1);
	log(magnitudeMat, magnitudeMat);
	nomalize(magnitudeMat, magnitudeMat, 0, 1, CV_MINMAX);
	magnitudeMat.convertTo(magnitudeMat, CV_8UC1, 255, 0);
	imshow("magnitudeMat", magnitudeMat);
	//频域中心移动
	int cx = magnitudeMat.cols / 2;
	int cy = magnitudeMat.rows / 2;
	Mat tmp;
	Mat q0(magnitudeMat, Rect(0, 0, cx, cy));
	Mat q1(magnitudeMat, Rect(cx, 0, cx, cy));
	Mat q2(magnitudeMat, Rect(0, cy, cx, cy));
	Mat q3(magnitudeMat, Rect(cx, cy, cx, cy));
	//交换象限
	q0.copyTo(tmp);
	q3.copyTo(q0);
	temp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
	imshow("magnitudeMat", magnitudeMat);
	//倾斜角检测
	Mat binaryMagnMat;
	threshold(magnitudeMat, binaryMagnMat, 133, 255, CV_THRESH_BINARY);
	imshow("binaryMagnMat", binaryMagnMat);
	std::vector<Vec2f> lines;
	binaryMagnMat.convertTo(binaryMagnMat, CV_8UC1, 255, 0);
	HoughLines(binaryMagnMat, lines, 1, CV_PI/180, 100, 0, 0);
	Mat houghMat(binaryMagnMat.size(), CV_8UC3);
	for (size_t i = 0; i < lines.size(); ++i)
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000*(-b)); 
		pt1.y = cvRound(y0 + 1000*(a));
		pt2.x = cvRound(x0 - 1000*(-b)); 
		pt2.y = cvRound(y0 - 1000*(a));  
		line(houghMat, pt1, pt2, Scalar(0, 0, 255), 3, CV_AA);
	}
	imshow("houghMat", houghMat);
	float theta = 0;
	for (size_t i = 0; i < lines.size(); ++i)
	{
		float thetaTemp = lines[i][1]*180/CV_PI;
		if (thetaTemp > 0 && thetaTemp < 90)
		{
			theta = thetaTemp;
			break;
		}
	}
	float angelT = nRows*tan(theta/180*CV_PI)/nCols;
	theta = atan(angelT)*180/CV_PI;
	//仿射变换矫正
	Point2f centerPoint = Point2f(nCols/2, nRows/2);
	double scale = 1;
	Mat warpMat = getRotationMatrix2D(centerPoint, theta, scale);
	Mat resultImage(srcImage.size(), srcImage.type());
	warpAffine(srcImage, resultImage, warpMat, resultImage.size());
	imshow("resultImage", resultImage);
	return 0;
}

//图像差分操作
void diffOperation(const Mat srcImage, Mat& edgeXImage, Mat& edgeYImage)
{
	Mat tempImage = srcImage.clone();
	int nRows = tempImage.rows;
	int nCols = tempImage.cols;
	for (int i = 0; i < nRows - 1; ++i)
	{
		for (int j = 0; j < nCols - 1; ++j)
		{
			edgeXImage.at<uchar>(i, j) = abs(tempImage.at<uchar>(i+1, j) - tempImage.at<uchar>(i, j));
			edgeYImage.at<uchar>(i, j) = abs(tempImage.at<uchar>(i, j+1) - tempImage.at<uchar>(i, j));
		}
	}
}
int main(int argc, char const *argv[])
{
	Mat srcImage = imread("..\\images\\building.png", 0);
	if ( !srcImage.data)
	{
		return -1;
	}
	imshow("srcImage", srcImage);
	Mat edgeXImage(srcImage.size(), srcImage.type());
	Mat edgeYImage(srcImage.size(), srcImage.type());
	diffOperation(srcImage, edgeXImage, edgeYImage);
	Mat edgeImage(srcImage.size(), srcImage.type());
	addWeighted(edgeXImage, 0.5, edgeYImage, 0.5, 0.0, edgeImage);
	waitkey(0);
	return 0;
}
//非极大值抑制sobel边缘实现
bool SobelVerEdge(Mat srcImage, Mat& resultImage)
{
	CV_Assert(srcImage.channels() == 1);
	srcImage.convertTo(srcImage, CV_32FC1);
	Mat sobelx = (Mat_<float>(3, 3) << -0.125, 0, 0.125
		-0.25, 0, 0.25,
		-0.125, 0, 0.125);
	Mat ConResMat;
	filter2D(srcImage, ConResMat, srcImage.type(), sobelx);
	Mat graMagMat;
	multiply(ConResMat, ConResMat, graMagMat);
	int scaleVal = 4;
	double thresh = scaleVal * mean(graMagMat).val[0];//mean()均值
	Mat resultTempMat = Mat::zeros(graMagMat.size(), graMagMat.type());
	float* pDataMag = (float*)graMagMat.data;
	float* pDataRes = (float*)resultTempMat.data;
	const int nRows = ConResMat.rows;
	const int nCols = ConResMat.cols;
	for (int i = 1; i != nRows - 1; ++i)
	{
		for (int j = 1; j != nCols - 1; ++j)
		{
			bool b1 = (pDataMag[i*nCols + j]) > (pDataMag[i*nCols + j - 1]);
			bool b2 = (pDataMag[i*nCols + j]) > (pDataMag[i*nCols + j + 1]);
			bool b3 = (pDataMag[i*nCols + j]) > (pDataMag[(i - 1)*nCols + j]);
			bool b4 = (pDataMag[i*nCols + j]) > (pDataMag[(i + 1)*nCols + j]);
			pDataRes[i*nCols + j] = 255 * ((pDataMag[i*nCols + j] > thresh) && ((b1 && b2) || (b3 && b4)));
		}
	}
	resultTempMat.convertTo(resultTempMat, CV_8UC1);
	resultImage = resultTempMat.clone();
	return true;
}
//图像直接卷积实现sobel
bool sobelEdge(const Mat& srcImage, Mat& resultImage, uchar threshold)
{
	CV_Assert(srcImage.channels() == 1);
	Mat sobelx = (Mat_<double>(3, 3) << 1, 0,
		-1, 2, 0, -2, 1, 0, -1);
	Mat sobely = (Mat_<double>(3, 3) << 1, 2, 1,
		0, 0, 0, -1, -2, -1);
	resultImage = Mat::zeros(srcImage.rows - 2, srcImage.cols - 2, srcImage.type());
	double edgeX = 0;
	double edgeY = 0;
	double graMag = 0;
	for (int k = 0; k < srcImage.rows - 1; ++k)
	{
		for (int n = 0; n < srcImage.rows - 1; ++n)
		{
			edgeX = 0;
			edgeY = 0;
			for (int i = -1; i <= 1; ++i)
			{
				for (int j = -1; j <= 1; ++j)
				{
					edgeX += srcImage.at<uchar>(k + i, n + j) * sobelx.at<double>(1 + i, 1 + j);
					edgeY += srcImage.at<uchar>(k + i, n + j) * sobely.at<double>(1 + i, 1 + j);
				}
			}
			graMag = sqrt(pow(edgeY, 2) + pow(edgeY, 2));
			resultImage.at<uchar>(k-1, n-1) = ((graMag > threshold)?255:0);
		}
	}
	return true;
}
//图像卷积实现sobel 非极大值抑制
bool sobelOptaEdge(const Mat& srcImage, Mat& resultImage, int flag)
{
	CV_Assert(srcImage.channels() == 1);
	Mat sobelx = (Mat_<double>(3, 3) << 1, 0,
		-1, 2, 0, -2, 1, 0, -1);
	Mat sobely = (Mat_<double>(3, 3) << 1, 2, 1,
		0, 0, 0, -1, -2, -1);
	//计算水平与垂直卷积
	Mat edgeX, edgeY;
	filter2D(srcImage, edgeX, edgeY, CV_32FC1, sobelx);
	filter2D(srcImage, edgeY, edgeY, CV_32FC1, sobely);
	//根据传入参数确定计算水平或垂直边缘
	int paraX = 0;
	int paraY = 0;
	switch(flag)
	{
		case 0: paraX = 1;
				paraY = 0;
				break;
		case 1: paraX = 0;
				paraY = 1;
				break;
		case 2: paraX = 1;
				paraY = 1;
				break;
		default : break;
	}
	edgeX = abs(edgeX);
	edgeY = abs(edgeY);
	Mat graMagMat = paraX * edgeX.mul(edgeX) + paraY * edgeY.mul(edgeY);
	//计算阀值
	int scaleVal = 4;
	double thresh = scaleVal * mean(graMagMat).val[0];
	resultImage = Mat::zeros(srcImage.size(), srcImage.type());
	for (int i = 0; i < srcImage.rows - 1; ++i)
	{
		float* pDataEdgeX = edgeX.ptr<float>(i);
		float* pDataEdgeY = edgeY.ptr<float>(i);
		float* pDataMag = graMagMat.ptr<float>(i);
		for (int j = 0; j < srcImage.clos - 1; ++j)
		{
			//判断当前邻域梯度是否大于阀值与大于水平或垂直梯度
			if (pDataMag[j] > thresh && 
				( ( pDataEdgeX[j] > paraX*pDataEdgeY[j] &&
					pDataMag[j] > pDataMag[j-1] &&
					pDataMag[j] > pDataMag[j+1]
				  ) ||
				  ( pDataEdgeY[j] > paraY*pDataEdgeX[j] &&
				  	pDataMag[j] > pDataMag[j-1] &&
				  	pDataMag[j] > pDataMag[j+1]
				  )
				)
			   )
			{
				resultImage.at<uchar>(i, j) = 255;
			}
		}
	}
	return true;
}
//opencv 自带库图像Sobel边缘计算
int main(int argc, char const *argv[])
{
	Mat srcImage = imread("..\\images\\building.png");
	if ( !srcImages.data )
	{
		return -1;
	}
	Mat srcGray;
	cvtColor(srcImage, srcGray, CV_BGR2GRAY);
	imshow("srcGray", srcGray);
	Mat edgeMat, edgeXMat, edgeYMat;
	Sobel(srcGray, edgeXMat, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	Sobel(srcGray, edgeYMat, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(edgeXMat, edgeXMat);
	convertScaleAbs(edgeYMat, edgeYMat);
	addWeighted(edgeXMat, 0.5, edgeXMat, 0.5, 0, edgeMat);
	imshow("edgeMat", edgeMat);
	//定义Scharr边缘图像
	Mat edgeMatS, edgeXMatS, edgeYMatS;
	Scharr(srcGray, edgeXMatS, CV_16S, 1, 0, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(edgeXMatS, edgeXMatS);
	Scharr(srcGray, edgeYMatS, CV_16S, 1, 0, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(edgeYMatS, edgeYMatS);
	addWeighted(edgeXMatS, 0.5, edgeYMatS, 0.5, 0, edgeMatS);
	imshow("edgeMatS", edgeMatS);
	return 0;
}
//Laplace边缘检测
int main(int argc, char const *argv[])
{
	Mat srcImage = imread("..\\images\\building.png", 0);
	if ( !srcImages.data )
	{
		return -1;
	}
	//高斯平滑
	GaussianBlur(srcImage, srcImage, Size(3, 3), 0, 0, BORDER_DEFAULT);
	Mat dstImage;
	Laplacian(srcImage, dstImage, CV_16S, 3);
	convertScaleAbs(dstImage, dstImage);
	waitkey(0);
	return 0;
}
//canny边缘检测
int main(int argc, char const *argv[])
{
	Mat srcImage = imread("..\\images\\building.png", 0);
	if ( !srcImages.data )
	{
		return -1;
	}
	int edgeThresh = 50
	Mat dstImage;
	Canny(srcImage, dstImage, edgeThresh, edgeThresh * 3, 3);
	imshow("dstImage", dstImage);
	waitkey(0);
	return 0;
}
//canny边缘检测的原理与实现
//1，消除噪声
Mat src, image;
GaussianBlur(src, image, Size(3, 3), 1.5);
//2，计算梯度幅值与方向
Mat magX = Mat(src.rows, src.cols, CV_32F);
Mat magY = Mat(src.rows, src.cols, CV_32F);
Sobel(image, magX, CV_32F, 1, 0, 3);
Sobel(image, magY, CV_32F, 0, 1, 3);
Mat slopes = Mat(image.rows, image.cols, CV_32F);
divide(magY, magX, slopes);
Mat sum = Mat(image.rows, image.clos, CV_64F);
Mat prodX = Mat(image.rows, image.clos, CV_64F);
Mat prodY = Mat(image.rows, image.clos, CV_64F);
multiply(magX, magX, prodX);
multiply(magY, magY, prodY);
sum = prodX + prodY;
sqrt(sum, sum);
Mat magnitude = sum.clone();
//3，非极大值抑制
void nonMaximumSuppression(Mat &magnitudeImage, Mat &directionImage)
{
	Mat checkImage = Mat(magnitudeImage.rows, magnitudeImage.cols, CV_8U);
	MatIterator_<float>itMag = magnitudeImage.begin<float>();
	MatIterator_<float>itDirection = directionImage.begin<float>();
	MatIterator_<unsigned char>itRet = checkImage.begin<unsigned char>();
	MatIterator_<float>itEnd = magnitudeImage.end<float>();
	for (; itMag != itEnd; ++itMag, ++itDirection, ++itRet)
	{
		const Point pos = itRet.pos();
		float currentDirection = atan(*itDirection) * (180 / 3.142);
		while(currentDirection < 0)
			currentDirection += 180;
		*itDirection = currentDirection;
		if (currentDirection > 22.5 && currentDirection <= 67.5)
		{
			if (pos.y > 0 && pos.x > 0 && *itMag <= magnitudeImage.at<float>(pos.y - 1, pos.x - 1))
			{
				magnitudeImage.at<float>(pos.y, pos.x) = 0;
			}
			if (pos.y < magnitudeImage.rows -1 && pos.x < magnitudeImage.cols - 1 && *itMag <= magnitudeImage.at<float>(pos.y + 1, pos.x + 1))
			{
				magnitudeImage.at<float>(pos.y, pos.x) = 0;
			}
		}else if (currentDirection >67.5 && currentDirection <= 112.5)
		{
			if (pos.y > 0 && *itMag <= magnitudeImage.at<float>(pos.y - 1, pos.x))
			{
				magnitudeImage.at<float>(pos.y, pos.x) = 0;
			}
			if (pos.y < magnitudeImage.rows -1 && *itMag <= magnitudeImage.at<float>(pos.y + 1, pos.x))
			{
				magnitudeImage.at<float>(pos.y, pos.x) = 0;
			}
		}else if (currentDirection >112.5 && currentDirection <= 157.5)
		{
			if (pos.y > 0 && pos.x < magnitudeImage.cols - 1 && *itMag <= magnitudeImage.at<float>(pos.y - 1, pos.x + 1))
			{
				magnitudeImage.at<float>(pos.y, pos.x) = 0;
			}
			if (pos.y < magnitudeImage.rows -1 && pos.x > 0 && *itMag <= magnitudeImage.at<float>(pos.y + 1, pos.x - 1))
			{
				magnitudeImage.at<float>(pos.y, pos.x) = 0;
			}
		}else
		{
			if (pos.x > 0 && *itMag <= magnitudeImage.at<float>(pos.y, pos.x - 1))
			{
				magnitudeImage.at<float>(pos.y, pos.x) = 0;
			}
			if (pos.x < magnitudeImage.cols -1 && *itMag <= magnitudeImage.at<float>(pos.y, pos.x + 1))
			{
				magnitudeImage.at<float>(pos.y, pos.x) = 0;
			}
		}
	}
}
//4，滞后阀值边缘连接
void followEdges(int x, int y, Mat &magnitude, int tUpper, int tLower, Mat &edges){//边缘连接
	edges.at<float>(y, x) = 255;
	for (int i = -1; i < 2; ++i)
	{
		for (int j = -1; j < 2; ++j)
		{
			if ( (i != 0) && (j != 0) && (x+i >= 0) && (y+j>=0) && (x+i <= magnitude.cols) && (y+j <=magnitude.rows) )
			{
				if ((magnitude.at<float>(y+j, x+i) > tLower) && (edges.at<float>(y+j, x+i) != 255))
				{
					followEdges(x+i, y+j, magnitude, tUpper, tLower, edges);
				}
			}
		}
	}
}
void edgeDetect(Mat &magnitude, int tUpper, int tLower, Mat &edges)
{
	int rows = magnitude.rows;
	int cols = magnitude.cols;
	edges = Mat(magnitude.size(), CV_32F, 0.0);
	for (int x = 0; x < cols; ++x)
	{
		for (int y = 0; y < rows; ++y)
		{
			if (magnitude.at<float>(y, x) >= tUpper)
			{
				followEdges(x, y, magnitude, tUpper, tLower, edges);
			}
		}
	}
}

//Marr-Hildreth边缘检测
void marrEdge(const Mat src, Mat &result, int kerValue, double delta)
{
	Mat kernel;//计算LoG算子
	int kerLen = kerValue/2;
	kernel = Mat_<double>(kerValue, kerValue);
	for (int i = -kerLen; i < kerLen; ++i)
	{
		for (int j = -kerLen; j < kerLen; ++j)
		{
			kernel.at<double>(i+kerLen, j+kerLen) = 
				exp(-((pow(j,2) + pow(i, 2))/(pow(delta, 2)*2) )) * 
				(((pow(j, 2) + pow(i, 2) - 2*pow(delta, 2))/(2*pow(delta, 4))));
		}
	}
	int kerOffset = kerValue / 2;
	Mat laplacian = (Mat_<double>(src.rows - kerOffset*2, src.cols - kerOffset*2));
	result = Mat::zeros(src.rows - kerOffset*2,  src.cols - kerOffset*2, src.type());
	double sumLaplacian;
	for (int i = 0; i < src.rows-kerOffset; ++i)
	{
		for (int j = 0; j < src.cols-kerOffset; ++j)
		{
			sumLaplacian = 0;
			for (int k = -kerOffset; k < kerOffset; ++k)
			{
				for (int m = -kerOffset; m < kerOffset; ++m)
				{
					sumLaplacian += src.at<uchar>(i+k, j+m) * kernel.at<double>(kerOffset+k, kerOffset+m);
				}
			}
			laplacian.at<double>(i - kerOffset, j - kerOffset) = sumLaplacian;
		}
	}

	for (int y = 0; y < result.rows - 1; ++y)
	{
		for (int y = 0; y < result.cols - 1; ++y)
		{
			result.at<uchar>(y, x) = 0;
			if (laplacian.at<double>(y - 1, x) * laplacian.at<double>(y + 1, x) < 0)
			{
				result.at<uchar>(y, x) = 255;
			}
			if (laplacian.at<double>(y, x-1) * laplacian.at<double>(y, x+1) < 0)
			{
				result.at<uchar>(y, x) = 255;
			}
			if (laplacian.at<double>(y + 1, x-1) * laplacian.at<double>(y - 1, x+1) < 0)
			{
				result.at<uchar>(y, x) = 255;
			}
			if (laplacian.at<double>(y - 1, x-1) * laplacian.at<double>(y + 1, x+1) < 0)
			{
				result.at<uchar>(y, x) = 255;
			}
		}
	}
}
//霍夫变换线检测
int main(int argc, char const *argv[])
{
	Mat srcImage = imread("..\\images\\building.png", 0);
	if ( !srcImage.data)
	{
		return -1;
	}
	
	Mat edgeMat, houghMat;
	Canny(srcImage, edgeMat, 50, 200, 3);
	cvtColor(edgeMat, houghMat, CV_GRAY2BGR);
	#if
	//标准的霍夫变换
	vector<Vec2f> lines;
	HoughLines(edgeMat, lines, 1, CV_PI/180, 100, 0, 0);
	for (size_t i = 0; i < lines.size(); ++i)
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000*(-b));
		pt1.y = cvRound(x0 + 1000*(a));
		pt2.x = cvRound(x0 - 1000*(-b));
		pt2.y = cvRound(x0 - 1000*(a));
		line(houghMat, pt1, pt2, Scalar(0, 0, 255), 3, CV_AA);
	}
	#else
	//统计概率的霍夫变换
	vector<Vec4i> lines;
	HoughLinesP(edgeMat, lines, 1, CV_PI/180, 50, 50, 10);
	for (size_t i = 0; i < lines.size(); ++i)
	{
		Vec4i l = lines[i];
		line(houghMat, Point(l[0],l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, CV_AA);
	}
	#endif
	imshow("srcImage", srcImage);
	imshow("houghMat", houghMat)
	return 0;
}
//CUDA初始化完整代码
#include <stdio.h>
#include <cuda_runtime.h>
void printDeviceProp(const cudaDeviceProp &prop)
{
	printf("Device Name: %s.\n", prop.name);
	printf("totalGlobalMem: %d.\n", prop.totalGlobalMem);
	printf("sharedMemPerBlock: %d.\n", prop.sharedMemPerBlock);
	printf("regsPerBlock: %d.\n", prop.regsPerBlock);
	printf("warpSize: %d.\n", prop.warpSize);
	printf("memPitch: %d.\n", prop.memPitch);
	printf("maxThreadsPerBlock: %d.\n", prop.maxThreadsPerBlock);
	printf("maxThreadsDim[0-2]: %d %d %d.\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prp.maxThreadsDim[2]);
	printf("maxGridSize[0-2]: %d %d %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
	printf("totalConstMem: %d.\n", prop.totalConstMem);
	printf("major.minor: %d %d.\n", prop.major, prop.minor);
	printf("clockRate: %d.\n", prop.clockRate);
	printf("textureAlignment: %d.\n", prop.textureAlignment);
	printf("deviceOverlap: %d.\n", prop.deviceOverlap);
	printf("multiProcessCount: %d.\n", prop.multiProcessCount);
}
bool InitCUDA()
{
	//used to count the device numbers
	int count;

	//get the cuda device count
	cudaGetDeviceCount(&count);
	if (count == 0)
	{
		fprintf(stderr, "There is no device.\n");
		return false;
	}

	//find the device >= 1.x
	int i;
	for (int i = 0; i < count; ++i)
	{
		cudaDeviceProp prop;
		if (cudaGetDeviceProperties(&prop, i) == cudaSucess)
		{
			if (prop.major >= 1)
			{
				printDeviceProp(prop);
				break;
			}
		}
	}

	//if can't find the device
	if (i == count)
	{
		fprintf(stderr, "There is no device supporting CUDA 1.x.\n",);
		return false;
	}

	//set cuda device 
	cudaSetDevice(i);
	return true;
}
int main(int argc, char const *argv[])
{
	if (InitCUDA())
	{
		printf("CUDA is initialized\n");
	}
	return 0;
	return 0;
}

#define DATA_SIZE 1048576  //1M
#define THREAD_NUM 512   	//thread num

__global__ void add(int a, int b, int *c)
{
	*c = a + b;
}
int main(){
	int c; 
	int *dev_c;
	cudaMalloc((void**)&dev_c, sizeof(int));
	add<<<1,1>>>(2, 7, dev_c);
	cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
	cout <<"2+7= " << c<< endl;
	cudaFree(dev_c);
	return;
}

#include "../common/book.h"
#include <iostream>

__global__ void kernel(void){

}
__global__ void add(int a, int b, int *c){
	*c = a + b;
}

int main(void)
{
	//kernel<<<1, 1>>>();
	int c;
	int *dev_c;
	HANDLE_ERROR(cudaMalloc((void**)&dev_c, sizeof(int)));
	add<<<1, 1>>>(2, 7, dev_c);
	HANDLE_ERROR(cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost));
	printf("2 + 7 = %d\n", c);
	cudaFree(dev_c);
	return 0;
}

int count;
HANDLE_ERROR(cudaGetDeviceCount(&count));
struct cudaDeviceProp{
	char name[256];
	size_t totalGlobalMem;
	size_t sharedMemPerBlock;
	int regsPerBlock;
	int warpSize;
	size_t memPitch;
	int maxThreadsPerBlock;
	int maxThreadsDim;
	int MaxGridSize[3];
	size_t totalConstMem;
	int major;
	int minor;
	int clockRate;
	size_t textureAlignment;
	int deviceOverlap;
	int multiProcessCount;
	int kernelExecTimeoutEnabled;
	int integrated;
	int canMapHostMemory;
	int computeMode;
	int maxTexture1D
	int maxTexture3D[3];
	int maxTexture2DArray[3];
	int concurrentKernels;
	int maxTexture2D[2];
};

#include "../common/book.h"
int main(void){
	cudaDeviceProp prop;
	int count;
	HANDLE_ERROR(cudaGetDeviceCount( &count ));
	for (int i = 0; i < count; ++i)
	{
		HANDLE_ERROR(cudaGetDeviceProperties( &prop, i));
		printf("   ---General Information for device %d ---\n", i);
		printf("Name:　%d\n", prop.name);
		printf("Compute capability: %d.%d\n", prop.major, prop.minor);
		printf("Clock rate: %d\n", prop.clockRate);
		printf("Device copy overlap: ");
		if (prop.deviceOverlap)
		{
			printf("Enabled\n");
		}else{
			printf("Disabled\n");
		}
		printf("kernel execition timeout: \n");
		if(prop.kernelExecTimeoutEnabled){
			printf("Enabled\n");
		}else{
			printf("Disabled\n");
		}
		printf("  --- Memory Information for device %d ---\n", i);
		printf("Total global mem: %ld\n", prop.totalGlobalMem);
		printf("Total constant Mem: %ld\n", prop.totalConstMem);
		printf("Max mem pitch: %ld\n", prop.memPitch);
		printf("Texture Alignment: %ld\n", prop.textureAlignment);
		printf("  --- MP Information for device %d ---\n", i);
		printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
		printf("Shared mem per mp: %ld\n", prop.sharedMemPerBlock);
		printf("Registers per mp: %d\n", prop.regsPerBlock);
		printf("Threads in warp: %d\n", prop.warpSize);
		printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
		printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		printf("\n");
	}
}

cudaDeviceProp prop;
memset(&prop, 0, sizeof(cudaDeviceProp));
prop.major = 1;
prop.minor = 3;

#include "../common/book.h"

int main(void){
	cudaDeviceProp prop;
	int dev;

	HANDLE_ERROR(cudaGetDevice( &dev ));
	printf("ID of current CUDA device: %d\n", dev);

	memset(&prop, 0 ,sizeof(cudaDeviceProp));
	prop.major = 1;
	prop.minor = 3;
	HANDLE_ERROR(cudaChooseDevice(&dev, &prop));
	printf("ID of CUDA device closest to revision 1.3: %d\n", dev);
}   

#include "../common/book.h"
#define N 10
void add(int *a, int *b, int *c){
	int tid = 0;
	while(tid<N){
		c[tid] = a[tid] + b[tid];
		tid += 1;
	}
}
int main(void){
	int a[N], b[N], c[N];
	for (int i = 0; i < N; ++i)
	{
		a[i] = -i;
		b[i] = i*i;
	}
	add(a, b, c);
	for (int i = 0; i < N; ++i)
	{
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}
}

#include "../common/book.h"
#define N 10

int main(void){
	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c;
	//GPU上分配内存
	HANDLE_ERROR( cudaMalloc( (void**)&dev_a, N * sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_b, N * sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_c, N * sizeof(int) ) );

	//在CPU上为数组a和b赋值
	for (int i = 0; i < N; ++i)
	{
		a[i] = -i;
		b[i] = i*i;
	}
	//将数组a和b复制到GPU
	HANDLE_ERROR(cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice));
	add<<<N,1>>>(dev_a, dev_b,dev_c);

	//将数组c从GPU复制到CPU
	HANDLE_ERROR(cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost));
	//显示结果
	for (int i = 0; i < N; ++i)
	{
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}
	//释放在GPU上分配的内存
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	return 0;
} 

__global__ void add(int *a, int *b, int *c){
	int tid = blockIdx.x;    //计算该索引处的数据
	if (tid<N)
	{
		c[tid] = a[tid] + b[tid];
	}
}

int main(void){
	CPUBitmap bitmap(DIM, DIM);
	unsigned char *ptr = bitmap.get_ptr();

	kernel(ptr);
	bitmap.display_and_exit();
}

void kernel(unsigned char *ptr){
	for (int y = 0; y < DIM; ++y)
	{
		for (int x = 0; x < DIM; ++x)
		{
			int offset = x + y*DIM;
			int juliaValue = julia(x, y);
			ptr[offset*4 + 0] = 255*juliaValue;
			ptr[offset*4 + 1] = 0;
			ptr[offset*4 + 2] = 0;
			ptr[offset*4 + 3] = 255;
		}
	}
}

#include "../common/book.h"
#include "../common/cpu_bitmap.h"

#define DIM 100
struct cuComplex
{
	float r;
	float i;
	cuComplex(float a, float b):r(a),i(b){}
	__device__ float magnitude2(void) {
		return r*r + i*i;
	}
	__device__ cuComplex operator*(const cuComplex& a){
		return cuComplex(r*a.r - i*a.r, i*a.r + r*a.i);
	}
	__device__ cuComplex operator+(const cuComplex& a){
		return cuComplex(r+a.r, i+a.r);
	}
};
__device__ int julia(int x, int y){
	const float scale = 1.5;
	float jx = scale * (float)(DIM/2 - x)/(DIM/2);
	float jy = scale * (float)(DIM/2 - y)/(DIM/2);

	cuComplex c(-0.8, 0.156);
	cuComplex a(jx, jy);

	int i = 0;
	for (int i = 0; i < 200; ++i)
	{
		a = a*a + c;
		if (a.magnitude2()>100)
		{
			return 0;
		}
	}
	return 1;
}
__global__ void kernel(unsigned char *ptr){
	//将threadIdx/BlockIdx映射到像素位置
	int x = blockIdx.x;
	int y = blockIdx.y;
	int offset = x + y*gridMim.x;
	//现在计算这个位置上的值
	int juliaValue = julia(x, y);
	ptr[offset*4 + 0] = 255*juliaValue;
	ptr[offset*4 + 1] = 0;
	ptr[offset*4 + 2] = 0;
	ptr[offset*4 + 3] = 255;
}

int main(void){
	CPUBitmap bitmap(DIM, DIM);
	unsigned char *dev_bitmap;

	HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));

	dim3 grid(DIM, DIM);
	kernel<<<grid,1>>>(dev_bitmap);

	HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));

	bitmap.display_and_exit();

	HANDLE_ERROR(cudaFree(dev_bitmap));
}


//矢量求和
#include "../common/book.h"
#define N 10

int main(void){
	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c;
	//GPU上分配内存
	HANDLE_ERROR( cudaMalloc( (void**)&dev_a, N * sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_b, N * sizeof(int) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_c, N * sizeof(int) ) );

	//在CPU上为数组a和b赋值
	for (int i = 0; i < N; ++i)
	{
		a[i] = -i;
		b[i] = i*i;
	}
	//将数组a和b复制到GPU
	HANDLE_ERROR(cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice));
	add<<<1,N>>>(dev_a, dev_b,dev_c);
	//add<<<(N+127)/128, 128>>>(dev_a, dev_b,dev_c);
	//add<<<128, 128>>>(dev_a, dev_b,dev_c);

	//将数组c从GPU复制到CPU
	HANDLE_ERROR(cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost));
	//显示结果
	for (int i = 0; i < N; ++i)
	{
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}
	//释放在GPU上分配的内存
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	return 0;
} 

__global__ void add(int *a, int *b, int *c){
	int tid = threadIdx.x;    //计算该索引处的数据
	//int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid<N)
	{
		c[tid] = a[tid] + b[tid];
		tid += blockDim.x * gridDim.x;
	}
}

#include "../common/book.h"
#define imin(a,b) (a<b?a:b)

const int N = 33*1024;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin(32, (N+threadsPerBlock - 1)/threadsPerBlock);

__global__ void dot(float *a, float*b, float*c){
	__shared__ float cache[threadsPerBlock];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;
	float temp = 0;
	while(tid<N){
		temp += a[tid] * b[tid];
		tid += blockDim.x * gridDim.x;
	}
	cache[cacheIndex] = temp;
	__syncthreads();

	int i = blockDim.x / 2;
	while(i != 0){
		if (cacheIndex < i)
		{
			cache[cacheIndex] += cache[cacheIndex + i];
		}
		__syncthreads();
		i /= 2;
	}
	if (cacheIndex == 0)
	{
		c[blockIdx.x] = cache[0];
	}
}
int main(void){
	float *a, *b, c, *partial_c;
	float *dev_a, *dev_b, *dev_partial_c;

	a = (float*)malloc( N*sizeof(float) );
	b = (float*)malloc( N*sizeof(float) );
	partial_c = (float*)malloc( N*sizeof(float) );

	HANDLE_ERROR( cudaMalloc((void**)&dev_a, N*sizeof(float)));
	HANDLE_ERROR( cudaMalloc((void**)&dev_b, N*sizeof(float)));

	for (int i = 0; i < N; ++i)
	{
		a[i] = i;
		b[i] = i*2;
	}

	HANDLE_ERROR( cudaMemcpy(dev_a, a, N*sizeof(float), cudaMemcpyHostToDevice) );
	HANDLE_ERROR( cudaMemcpy(dev_b, b, N*sizeof(float), cudaMemcpyHostToDevice) );
	dot<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);

	HANDLE_ERROR( cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost) );

	c = 0;
	for (int i = 0; i < blocksPerGrid; ++i)
	{
		c += partial_c[i];
	}

	#define sum_squares(x) (x*(x+1)*(2*x+1)/6)
	printf("Does GPU value %.6g = %.6g?\n", c, 2*sum_squares((float)(N-1)));

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_partial_c);

	free(a);
	free(b);
	free(partial_c); 
}

//颜色圆检测
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>

using namespace cv;

int main(){
	Mat srcImage = imread("..\\images\\circles.png");
	if (!srcImage.data)
		return -1;
	imshow("srcImage", srcImage);
	Mat resultImage = srcImage.clone();
	//中值滤波
	medianBlur(srcImage, srcImage, 3);
	//转换成HSV空间
	Mat hsvImage;
	cvtColor(srcImage, hsvImage, COLOR_BGR2HSV);
	imshow("hsvImage", hsvImage);
	//阀值化
	Mat lowTempMat;
	Mat upperTempMat;
	//低阀值限制
	inRange(hsvImage, Scalar(0, 100, 100), Scalar(10, 255, 255), lowTempMat);
	//高阀值限制
	inRange(hsvImage, Scalar(160, 100, 100), Scalar(179, 255, 255), upperTempMat);
	//颜色阀值限定合并
	Mat redTempMat;
	addWeighted(lowTempMat, 1.0, upperTempMat, 1.0, 0.0, redTempMat);
	//高斯平滑
	GaussianBlur(redTempMat, redTempMat, Size(9, 9), 2, 2)
	//霍夫变换
	std::vector<Vec3f> circles;
	HoughCircles(redTempMat, circles, CV_HOUGH_GRADIENT, 1, redTempMat.rows/8, 100, 200, 0, 0);
	if (circles.size() == 0)
	{
		return 1
	}
	for (int i = 0; i < circles.size(); ++i)
	{
		Point center(round(circles[i][0]), round(circles[i][1]));
		int radius = round(circles[i][2]);
		circles(resultImage, center, radius, Scalar(0, 255, 0), 5);
	}
	imshow("resultImage", resultImage);
	waitkey(0);
	return 0;
}
//车牌区域检测
using namespace cv;
using namespace std;

int main(){
	Mat srcImage = imread("..\\images\\circles.png");
	if (!srcImage.data)
		return -1;
	//转换成HSV
	Mat img_h, img_s, img_v, imghsv;
	vector<Mat> hsv_vec;
	cvtColor(srcImage, imghsv, CV_BGR2HSV);
	imshow("hsv", imghsv);
	//分割HSV通道
	split(imghsv, hsv_vec);
	img_h = hsv_vec[0];
	img_s = hsv_vec[1];
	img_v = hsv_vec[2];
	//转换通道数据类型
	img_h.convertTo(img_h, CV_32F);
	img_s.convertTo(img_s, CV_32F);
	img_v.convertTo(img_v, CV_32F);
	//计算每个通道的最大值
	double max_s, max_h, max_v;
	minMaxIdx(img_h, 0, &max_h);
	minMaxIdx(img_s, 0, &max_s);
	minMaxIdx(img_v, 0, &max_v);
	//各个通道归一化
	img_h /= max_h;
	img_s /= max_s;
	img_v /= max_v;
	//饱和度图像通道sobel边缘提取
	Mat resultImage;
	SobelVerEdge(img_s, resultImage);
	//HSV限定范围元素提取
	Mat bw_blue = ((img_h>0.45) & (img_h<0.75)&(img_s>0.15)&(img_v>0.25));
	int height = bw_blue.rows;
	int width = bw_blue.cols;
	Mat bw_blue_edge = Mat::zeros(bw_blue.size(), bw_blue.type());
	imshow("bw_blue", bw_blue);
	waitkey(0);
	//车牌疑似区域提取
	for (int k = 1; k != height - 2; ++k)
	{
		for (int l = 1; l != width - 2; ++l)
		{
			//义窗搜索
			Rect rct;
			rct.x = l -1;
			rct.y = k -1;
			rct.height = 3;
			rct.width = 3;
			//判断当前点属于边缘且颜色区域内至少包含1个像素
			if ((resultImage.at<uchar>(k, 1) == 255) && (countNonZero(bw_blue(rct))>=1))
			{
				bw_blue_edge.at<uchar>(k, 1) = 255;
			}
		}
	}
	//形态学闭操作
	Mat morph;
	morphologyEx(bw_blue_edge, morph, MORPH_CLOSE, Mat::ones(2, 25, CV_8UC1));
	Mat imshow5;
	resize(bw_blue_edge, imshow5, Size(), 1, 1);
	imshow("morphology_bw_blue_edge", imshow5);
	waitkey(0);

	//连通区域选取
	imshow("morph", morph);
	//求闭操作的连通域外轮廓
	vector<vector<Point>> region_contours;
	findContours(morph.clone(), region_contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	//候选轮廓筛选
	vector<Rect> candidates;
	vector<Mat> candidates_img;
	Mat result;
	for (size_t n = 0; n != region_contours.size(); ++n)
	{
		//去除高度/宽度不符合条件区域
		Rect rect = boundingRect(region_contours[n]);
		//计算区域非零像素点
		int sub = countNonZero(morph(rect));
		//非零像素点占的比例
		double ratio = double(sub)/rect.area();
		//宽高比
		double wh_ratio = double(rect.width)/rect.height;
		if (ratio>0.5 && wh_ratio>2 && wh_ratio<5 && rect.height>12 && rect.width>60)
		{
			Mat small = bw_blue_edge(rect);
			result = srcImage(rect);
			imshow("rect", result);
			waitkey(0);
		}
	}
}
//细化sobel边缘提取
bool SobelVerEdge(Mat srcImage, Mat &resultImage){
	CV_Assert(srcImage.channels() == 1);
	srcImage.convertTo(srcImage, CV_32FC1);
	//水平方向的sobel算子
	Mat sobelx = (Mat_<float>(3,3) <<
		-0.125, 0, 0.125,
		-25, 0, 0.25,
		-0.125, 0, 0.125);
	Mat ConResMat;
	//卷积计算
	filter2D(srcImage, ConResMat, srcImage.type(), sobelx);
	//计算梯度的幅度
	Mat graMagMat;
	multiply(ConResMat, ConResMat, graMagMat);
	//根据梯度幅值及参数设置阀值
	int scaleVal = 4;
	double thresh = scaleVal*mean(graMagMat).val[0];
	Mat resultTempMat = Mat::zeros(graMagMat.size(), graMagMat.type());
	float* pDataMag = (float*)graMagMat.data;
	float* pDataRes = (float*)resultTempMat.data;
	const int nRows = ConResMat.rows;
	const int nCols = ConResMat.cols;
	for (int i = 1; i < nRows - 1; ++i)
	{
		for (int j = 1; j < nCols - 1; ++j)
		{
			bool b1 = (pDataMag[i*nCols + j] > pDataMag[i*nCols + j - 1]);
			bool b2 = (pDataMag[i*nCols + j] > pDataMag[i*nCols + j + 1]);
			bool b3 = (pDataMag[i*nCols + j] > pDataMag[(i-1)*nCols + j ]);
			bool b4 = (pDataMag[i*nCols + j] > pDataMag[(i+1)*nCols + j ]);
			pDataRes[i*nCols + j] = 255 * ((pDataMag[i*nCols + j] > thresh) && ((b1 && b2) || (b3&&b4)));
		}
		resultTempMat.convertTo(resultTempMat, CV_8UC1);
		resultImage = resultTempMat.clone();
		return true;
	}
}
//车牌区域提取
using namespace cv;
using namespace std;

Mat getPlate(int width, int height, Mat srcGray){
	Mat result;
	//形态学梯度检测边缘
	morphologyEx(srcGray, result, MORPH_GRADIENT, Mat(1, 2, CV_8U, Scalar(1)));
	imshow("1result", result);
	//阀值化
	threshold(result, result, 255*(0.1), 255, THRESH_BINARY);
	imshow("2result", result);
	//水平方向闭运算
	if (width >= 400 && width < 600)
	{
		morphologyEx(result, result, MORPH_CLOSE, Mat(1, 25, CV_8U, Scalar(1)));
	}else if (width >= 200 && width < 300)
	{
		morphologyEx(result, result, MORPH_CLOSE, Mat(1, 20, CV_8U, Scalar(1)));
	}else if (width >= 600)
	{
		morphologyEx(result, result, MORPH_CLOSE, Mat(1, 28, CV_8U, Scalar(1)));
	}else{
		morphologyEx(result, result, MORPH_CLOSE, Mat(1, 15, CV_8U, Scalar(1)));
	}
	//垂直方向闭运算
	if (height >= 400 && height < 600)
	{
		morphologyEx(result, result, MORPH_CLOSE, Mat(8, 1, CV_8U, Scalar(1)));
	}else if (height >= 200 && height < 300)
	{
		morphologyEx(result, result, MORPH_CLOSE, Mat(6, 1, CV_8U, Scalar(1)));
	}else if (height >= 600)
	{
		morphologyEx(result, result, MORPH_CLOSE, Mat(10, 1, CV_8U, Scalar(1)));
	}else{
		morphologyEx(result, result, MORPH_CLOSE, Mat(4, 1, CV_8U, Scalar(1)));
	}
	return result;
}
int main(int argc, char const *argv[])
{
	Mat srcImage = imread("..\\images\\car.jpg");
	if ( !srcImage.data)
	{
		return 1;
	}
	Mat srcGray;
	cvtColor(srcImage, srcGray, CV_BGR2GRAY);
	Mat result = getPlate(400, 300, srcGray);
	//连通域检测
	vector< vector<Point> > blue_contours;
	vector<Rect> blue_rect;
	findContours(result.clone(), blue_contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	//连通域遍历，车牌目标提取
	for (size_t i = 0; i != blue_contours.size(); ++i)
	{
		Rect rect = boundingRect(blue_contours[i]);
		double wh_ratio = double(rect.width) / rect.height;
		int sub = countNonZero(result(rect));
		double ratio = double(sub) / rect.area();
		if (wh_ratio > 2 && wh_ratio < 8 && rect.height > 12 && rect.width >60 && ratio > 0.4)
		{
			blue_rect.push_back(rect);
			imshow("rect", srcGray(rect));
			waitkey(0);
		}
	}
	imshow("result", result);
	return 0;
}
//分水岭图像分割
Mat watershedSegment(Mat &srcImage, int &noOfSegments){
	Mat grayMat;
	Mat otsuMat;
	cvtColor(srcImage, grayMat, CV_BGR2GRAY);
	imshow("grayMat", grayMat);
	//阀值操作
	threshold(grayMat, otsuMat, 0, 255, CV_THRESH_BINARY_INV+CV_THRESH_OTSU);
	imshow("otsuMat", otsuMat);
	//形态学开操作
	morphologyEx(otsuMat, otsuMat, MORPH_OPEN, Mat::ones(9, 9, CV_8SC1), Point(4, 4), 2);
	imshow("Mor-openMat", otsuMat);
	//距离变换
	Mat disTranMat(otsuMat.rows, otsuMat.cols, CV_32FC1);
	distanceTransform(otsuMat, disTranMat, CV_DIST_L2, 3);
	//归一化
	nomalize(disTranMat, disTranMat, 0.0, 1, NORM_MINMAX);
	imshow("DisTranMat", disTranMat);
	//阀值化分割
	threshold(disTranMat, disTranMat, 0.1, 1, CV_THRESH_BINARY);
	//归一化统计图像到0~255
	nomalize(disTranMat, disTranMat, 0.0, 255.0, NORM_MINMAX);
	disTranMat.convertTo(disTranMat, CV_8UC1);
	imshow("TDisTranMat", disTranMat);
	//计算标记的分割块
	int i, j, compCount = 0;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(disTranMat, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	if (contours.empty())
	{
		return Mat();
	}
	Mat markers(disTranMat.size(), CV_32S);
	markers = Scalar::all(0);
	int idx = 0;
	for (; idx >= 0; idx = hierarchy[idx][0], compCount++)
	{
		drawContours(markers, contours, idx, Scalar::all(compCount+1), -1, 8, hierarchy, INT_MAX);
	}
	if (compCount == 0)
	{
		return Mat();
	}
	//计算算法的时间复杂度
	double t = (double)getTickCount();
	watershed(srcImage, markers);
	t = (double)getTickCount() - t;
	printf("execition time = %gms\n", t*1000./getTickFrequency());
	Mat wshed = displySegResult(markers, compCount);
	imshow("watershed transform", wshed);
	noOfSegments = compCount;
	return markers;
}
//分割合并
void segMerge(Mat &image, Mat &segments, int &numSeg){
	vector<Mat> samples;
	int newNumSeg = numSeg;
	for (int i = 0; i <= numSeg; ++i)
	{
		Mat sampleImage;
		samples.push_back(sampleImage);
	}
	for (int i = 0; i < segments.rows; ++i)
	{
		for (int j = 0; j < segments.cols; ++j)
		{
			int index = segments.at<int>(i, j);
			if (index >= 0 && index<numSeg)
			{
				samples[index].push_back(image(Rect(j, i, 1, 1)));
			}
		}
	}
	vector<MatND> hist_bases;
	Mat hsv_base;
	int h_bins = 35;
	int s_bins = 30;
	int histSize[] = {h_bins, s_bins};
	float h_ranges[] = {0, 256};
	float s_ranges[] = {0, 180};
	const float* ranges[] = {h_ranges, s_ranges};
	int channels[] = {0, 1}
	MatND hist_bases;
	for (int c = 1; c < numSeg; ++c)
	 {
	 	if (samples[c].dims>0)
	 	{
	 		cvtColor(samples[c], hsv_base,CV_BGR2HSV);
	 		calcHist(&hsv_base, 1, channels, Mat(), hist_base, 2, histSize, ranges, true, false);
	 		normalize(hist_base, hist_base, 0, 1, NORM_MINMAX, -1, Mat());
	 		hist_bases.push_back(hist_base);
	 	}else{
	 		hist_bases.push_back(MatND());
	 	}
	 	hist_base.release();
	 }
	 double similarity = 0;
	 vector<bool> mearged;
	 for (int k = 0; k < hist_base.size(); ++k)
	 {
	 	mearged.push_back(false);
	 }

}

//FoodFill图像分割
using namespace cv;
using namespace std;

//初始化参数
Mat image, gray, mask;
int ffillMode = 1;
int loDoff = 20, upDiff = 20;
int connectivity = 4;
int isColor = true;
bool useMask = false;
int newMaskVal = 255;
//鼠标响应函数
static void onMouse(int event, int x, int y, int, void*){
	if (event != CV_EVENT_LBUTTONDOWN)
	{
		return;
	}
	//floodfill参数设置
	Point seed = Point(x, y);
	int lo = ffillMode == 0?0:loDoff;
	int up = ffillMode == 0?0:upDiff;
	int flags = connectivity + (newMaskVal << 8) + (ffillMode == 1?CV_FLOODFILL_FIXED_RANGE:0);
	//颜色分量随机选取
	int b = (unsigned)theRNG()&255;
	int g = (unsigned)theRNG()&255;
	int r = (unsigned)theRNG()&255;
	Rect ccomp;
	//颜色选择
	Scalar newVal = isColor ? Scalar(b, g, r) : Scalar(r*0.299 + g*0.587 + b*0.114);
	Mat dst = isColor ? image : gray;
	int area;
	//根据标志位选择泛洪填充
	if (useMask)
	{
		//阀值化操作
		threshold(mask, mask, 1, 128, CV_THRESH_BINARY);
		area = floodfill(dst, mask, seed, newVal, &ccomp, Scalar(lo, lo, lo),  Scalar(up, up, up), flags);
		imshow("mask", mask);
	}else{
		area = floodfill(dst, seed, newVal, &ccomp, Scalar(lo, lo, lo),  Scalar(up, up, up), flags);
	}
	imshow("image", dst);
}
int main(){
	Mat srcImage = imread("..\\images\\sea.jpg");
	if(srcImage.empty()) return 0;
	srcImage.copyTo(image);
	cvtColor(srcImage, gray, CV_BGR2GRAY);
	mask.create(srcImage.rows + 2, srcImage.cols + 2, CV_8UC1);
	//鼠标响应回调函数
	namedWindow("image", 0);
	setMouseCallback("image", onMouse, 0);
	for(;;){
		imshow("image", isColor?image:gray);
		int c = waitkey(0);
		if ((c&255) == 27)
		{
			cout << "Exiting...\n";
			break;
		}
		if (c == "r")
		{
			cout << "Original image is restored\n";
			srcImage.copyTo(image);
			cvtColor(image, gray, CV_BGR2GRAY);
			mask = Scalar::all(0);
		}
	}
	return 0;
}
//均值漂移图像分割
static void MergeSeg(Mat& img, const Scalar& colorDiff = Scalar::all(1)){
	CV_Assert(!img.empty());
	RNG rng = theRNG();
	Mat mask(img.rows + 2, img.cols + 2, CV_8UC1, Scalar::all(0));
	for (int y = 0; y < img.rows; ++y)
	{
		for (int x = 0; x < img.cols; ++x)
		{
			Scalar newVal(rng(256), rng(256), rng(256));
			foodFill(img, mask, Point(x, y), newVal, 0, colorDiff, colorDiff);
		}
	}
}
int main(int argc, char const *argv[])
{
	Mat srcImag = imread("..\\images\\sea.jpg");
	if (srcImag.empty())
	{
		return -1;
	}
	int spatialRad = 20;
	int colorRad = 20;
	int maxPyrLevel = 6;
	Mat resImg;
	pyrMeanShiftFiltering(srcImag, resImg, spatialRad, colorRad, maxPyrLevel);
	MergeSeg(resImg, Scalar::all(2));
	imshow("src", srcImag);
	imshow("resImg", resImg);
	waitkey();
	return 0;
}

//Grabcut图像分割
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>
#include <sstream>
using namespace cv;
using namespace std;
//全局变量
Point point1, point2;
int drag = 0;
Rect rect;
Mat srcImage, roiImg;
bool select_flag = true;
Mat rectImg;
vector<Point>Pf, Pb;
//鼠标响应回调
void mouseHandler(int event,int x, int y,int flags, void* param){
	//左键按下
	if (event == CV_EVENT_LBUTTONDOWN && !drag)
	{
		point1 = Point(x, y);
		drag = 1;
	}
	//鼠标移动
	else if (event == CV_EVENT_MOUSEMOVE && drag)
	{
		Mat img1 = srcImage.clone();
		point2 = Point(x, y);
		rectangle(img1, point1, point2, CV_RGB(255, 0, 0), 3, 8, 0);
		imshow("image", img1);
	//鼠标抬起与拖拽标志
	}else if (event == CV_EVENT_FLAG_LBUTTONUP && drag)
	{
		point2 = Point(x, y);
		rect = Rect(point1.x, point1.y, x - point1.x, y - point2.y);
		drag = 0;
		roiImg = srcImage(rect);
	//右键按下
	}else if (event == CV_EVENT_RBUTTONDOWN)
	{
		select_flag = false;
		drag = 0;
		imshow("ROI", roiImg);
	}
}
int main(){
	srcImage = imread("..\\images\\sea.png");
	if (srcImage.empty())
	{
		return -1;
	}
	//定义前景与输出图像
	Mat srcImage2 = srcImage.clone();
	Mat foreground(srcImage.size(), CV_8UC3, Scalar(255, 255, 255));
	Mat result(srcImage.size(), CV_8UC1);
	//Grabcut分割前景与背景
	Mat fgMat, bgMat;
	namedWindow("srcImage", 1);
	imshow("srcImage", srcImage);
	//迭代次数
	int i = 20;
	cout << "20 iters" <<endl;
	//鼠标响应
	setMouseCallback("srcImage", mouseHandler, NULL);
	//选择区域有效，按下esc退出
	while((select_flag == true) && (waitkey(0) != 27)){
		//实现图割操作
		grabCut(srcImage, result, rect, bgMat, fgMat, i, GC_INIT_WITH_RECT);
		compare(result, GC_PR_FGD, result, CMP_EQ);
		srcImage.copyTo(foreground,result);
		imshow("foreground", foreground);
	}
	waitkey(0);
	return 0;
}
//肤色检测
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
int main(){
	Mat srcImage, resultMat;
	srcImage = imread("..\\images\\handl.jpg");
	if (srcImage.empty())
	{
		return -1;
	}
	//构建肤色颜色空间椭圆模型
	Mat skinMat = Mat::zeros(Size(256, 256), CV_8UC1);
	ellipse(skinMat, Point(113, 155.6), Size(23.4, 15.2), 43.0, 0.0, 360.0, Scalar(255, 255, 255), -1);
	//定义结构元素
	Mat struElmen = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	Mat YcrcbMat;
	Mat tempMat = Mat::zeros(srcImage.size(), CV_8UC1);
	cvtColor(srcImage, YcrcbMat, CV_BGR2YCrCb);
	//椭圆皮肤模型检测
	for (int i = 0; i < srcImage.rows; ++i)
	{
		uchar* p = (uchar*)tempMat.ptr<uchar>(i)
		Vec3b* ycrcb = (Vec3b*)YcrcbMat.ptr<Vec3b>(i);
		for (int j = 0; j < srcImage.cols; ++j)
		{
			if (skinMat.at<uchar>(ycrcb[j][1], ycrcb[j][2]) > 0)
			{
				p[j] = 255;
			}
		}
	}
	//形态学闭操作
	morphologyEx(tempMat, tempMat, MORPH_CLOSE, struElmen);
	//定义轮廓参数
	vector<vector<Point>> contours;
	vector<vector<Point>> resContours;
	vector<Vec4i> hierarchy;
	//查找连通域
	findContours(tempMat, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	//筛选伪轮廓
	for (size_t i = 0; i < contours.size(); ++i)
	{
		//判断区域面积
		if (fabs(contourArea(Mat(contours[i]))) > 1000)
		{
			resContours.push_back(contours[i]);
		}
		tempMat.setTo(0);
		//绘制轮廓
		drawContours(tempMat, resContours, -1, Scalar(255, 0, 0), CV_FILLED);
		srcImage.copyTo(resultMat, tempMat);
		imshow("srcImage", srcImage);
		imshow("resultMat", resultMat);
		waitkey(0);
		return 0;
	}

}
