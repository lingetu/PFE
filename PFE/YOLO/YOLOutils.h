#ifndef YOLOUTILS_H
#define YOLOUTILS_H

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace dnn;
using namespace std;

Net loadYOLOModel();
vector<Rect> detectObjects(Mat& frame, Net& net);
vector<String> getOutputsNames(const Net& net);

#endif // YOLOUTILS_H