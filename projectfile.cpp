#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>

using namespace cv;
using namespace cv::face;
using namespace std;

int main() {
    CascadeClassifier faceCascade;
    if (!faceCascade.load("haarcascade_frontalface_default.xml")) {
        cout << "Error loading Haar Cascade file!" << endl;
        return -1;
    }
    Ptr<LBPHFaceRecognizer> recognizer = LBPHFaceRecognizer::create();
    recognizer->read("face_trainer.yml");

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Error opening webcam!" << endl;
        return -1;
    }

    Mat frame, gray;
    vector<Rect> faces;

    while (true) {
        cap >> frame;
        if (frame.empty())
            break;


        cvtColor(frame, gray, COLOR_BGR2GRAY);

        
        faceCascade.detectMultiScale(
            gray,
            faces,
            1.2,
            5,
            0,
            Size(100, 100)
        );

        for (size_t i = 0; i < faces.size(); i++) {
            Mat faceROI = gray(faces[i]);

            int label;
            double confidence;

            
            recognizer->predict(faceROI, label, confidence);

            string name;
            if (confidence < 70) {
                name = "Person " + to_string(label);
            } else {
                name = "Unknown";
            }

            
            rectangle(frame, faces[i], Scalar(0, 255, 0), 2);

        
            putText(
                frame,
                name,
                Point(faces[i].x, faces[i].y - 10),
                FONT_HERSHEY_SIMPLEX,
                0.8,
                Scalar(0, 255, 0),
                2
            );
        }

        imshow("Real-Time Face Recognition", frame);

        if (waitKey(10) == 27)  
            break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
