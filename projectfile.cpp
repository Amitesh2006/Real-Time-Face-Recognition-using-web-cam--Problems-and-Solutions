#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>

using namespace cv;
using namespace cv::face;
using namespace std;

int main() {
    // Load Haar Cascade
    CascadeClassifier faceCascade;
    if (!faceCascade.load("haarcascade_frontalface_default.xml")) {
        cout << "Error loading Haar Cascade file!" << endl;
        return -1;
    }

    // Load trained face recognizer
    Ptr<LBPHFaceRecognizer> recognizer = LBPHFaceRecognizer::create();
    recognizer->read("face_trainer.yml");

    // Open webcam
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout <
