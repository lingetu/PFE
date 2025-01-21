#include "YOLOUtils.h"


Net loadYOLOModel() {

    String modelConfiguration = "yoloMod\\yolov3-tiny.cfg";
    String modelWeights = "yoloMod\\yolov3-tiny.weights";

    // Vérifiez si les fichiers existent
    std::ifstream configCheck(modelConfiguration);
    std::ifstream weightsCheck(modelWeights);

    if (!configCheck.is_open()) {
        std::cerr << "Error: Configuration file not found: " << modelConfiguration << std::endl;
        throw std::runtime_error("Configuration file not found");
    }

    if (!weightsCheck.is_open()) {
        std::cerr << "Error: Weights file not found: " << modelWeights << std::endl;
        throw std::runtime_error("Weights file not found");
    }

    Net net;

    try {
        net = readNetFromDarknet(modelConfiguration, modelWeights);
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(DNN_TARGET_CPU);
    } catch (const cv::Exception& e) {
        std::cerr << "Error loading YOLO model: " << e.what() << std::endl;
        throw;
    }
    return net;
}

vector<String> getOutputsNames(const Net& net) {
    static vector<String> names;
    if (names.empty()) {
        vector<int> outLayers = net.getUnconnectedOutLayers();
        vector<String> layersNames = net.getLayerNames();
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

vector<Rect> detectObjects(Mat& frame, Net& net) {
    Mat blob;
    blobFromImage(frame, blob, 1/255.0, Size(416, 416), Scalar(0,0,0), true, false);
    net.setInput(blob);
    vector<Mat> outs;
    net.forward(outs, getOutputsNames(net));

    vector<Rect> boxes;
    for (size_t i = 0; i < outs.size(); ++i) {
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > 0.5) {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }
    return boxes;
}