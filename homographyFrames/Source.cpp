#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/imgproc/imgproc.hpp"


#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>

using namespace cv;
using namespace std;


int main(int argc, const char *argv[]) {
	//open webcam to capture video from it
	int choice;
	cout << " 1. Save frames and homography matrices to file.\n";
	cout << " 2. Read in frames and homography matrices from file.\n";
	cin >> choice;
	//////////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////////
	// Choice 1: output
	if (choice == 1){
		cout << "You picked choice 1!";
		VideoCapture cap(0);
		if (!cap.isOpened())
			return -1;

		Mat frame1c; // will hold the i'th frame, color
		Mat frame1; // ^^, grayscale
		Mat frame2c; // will hold the (i+1)'th frame, color
		Mat frame2; // ^^, grayscale
		std::vector<Mat> theframes; // will hold all the frames
		int myvar;
		string Hmatrices; // will hold all values of all H matrices.
		std::stringstream s;  //used for converting the numerical values to a string
		// Will be stored as h11 h12 h13 h21 h22 h23 h31 h32 h33[\n]h(2ndmatrix)11 h(2ndmatrix)12....etc
		//for(;;){

		cap >> frame1c;
		cvWaitKey(100);
		cvtColor(frame1c, frame1, CV_RGB2GRAY);
		cvWaitKey(100);


		//detect keypoints on first frame
		int minHessian = 400;
		SurfFeatureDetector detector(minHessian);
		std::vector<KeyPoint> keypoints1, keypoints2;
		detector.detect(frame1, keypoints1);

		for (int i = 0; i<20; i++){
			// Capture the next frames
			cap >> frame2c;
			cvWaitKey(300);
			cvtColor(frame2c, frame2, CV_RGB2GRAY);

			//Detect points on the next image
			detector.detect(frame2, keypoints2);


			// If no keypoints are found (e.g. when ThinkPad webcam displays black screen initially),
			// then go to the next iteration of the For loop.
			if (keypoints1.size() == 0){
				frame1c = frame2c.clone();
				frame1 = frame2.clone();
				keypoints1 = keypoints2;
				continue;
			}

			// Calculate feature-vectors of the points
			SurfDescriptorExtractor extractor;
			Mat descriptors1, descriptors2;
			extractor.compute(frame1, keypoints1, descriptors1);
			extractor.compute(frame2, keypoints2, descriptors2);

			// Match points on the 2 successive images by comparing feature-vectors
			FlannBasedMatcher matcher;
			std::vector<DMatch> matches;
			matcher.match(descriptors1, descriptors2, matches);

			//Eliminate weaker matches
			double maxdist = 0;
			double mindist = 100;
			for (int j = 0; j < descriptors1.rows; j++){
				double dist = matches[j].distance;
				if (dist < mindist) mindist = dist;
				if (dist > maxdist) maxdist = dist;
			}

			//build the list of "good" matches
			std::vector<DMatch> goodmatches;
			for (int k = 0; k < descriptors1.rows; k++){
				if (matches[k].distance <= 3 * mindist){
					goodmatches.push_back(matches[k]);
				}
			}


			//Now compute homography matrix  between the stronger matches

			//-- Localize the object
			std::vector<Point2f> obj;
			std::vector<Point2f> scene;
			if (goodmatches.size() < 4){
				frame1c = frame2c.clone();
				frame1 = frame2.clone();
				keypoints1 = keypoints2;
				continue;
			}

			// If the frames have keypoints, save them to "theframes"
			theframes.push_back(frame1c.clone());


			unsigned int i;
			std::stringstream s;
			for (int l = 0; l < goodmatches.size(); l++){
				//-- Get the keypoints from the good matches
				obj.push_back(keypoints1[goodmatches[l].queryIdx].pt);
				scene.push_back(keypoints2[goodmatches[l].trainIdx].pt);
			}

			Mat Hmatrix;
			Hmatrix = findHomography(obj, scene, CV_RANSAC);

			// Save the values of the H matrix into the Hmatrices string
			s << Hmatrix.at<double>(0, 0) << " ";
			s << Hmatrix.at<double>(0, 1) << " ";
			s << Hmatrix.at<double>(0, 2) << " ";

			s << Hmatrix.at<double>(1, 0) << " ";
			s << Hmatrix.at<double>(1, 1) << " ";
			s << Hmatrix.at<double>(1, 2) << " ";

			s << Hmatrix.at<double>(2, 0) << " ";
			s << Hmatrix.at<double>(2, 1) << " ";
			s << Hmatrix.at<double>(2, 2) << "\n";
			Hmatrices = Hmatrices + s.str();

			frame1c = frame2c.clone();
			frame1 = frame2.clone();
			keypoints1 = keypoints2;

		}//end for loop

		theframes.push_back(frame1c.clone());

		ofstream savetofile;
		savetofile.open("hmatrices.txt");
		Hmatrices.pop_back();
		savetofile << Hmatrices;
		savetofile.close();

		// Save theframes to jpeg files:
		for (int m = 0; m<theframes.size(); m++){
			imwrite("frame" + std::to_string((long long)m) + ".jpg", theframes[m]);
		}

		return 0;
	}
	/////////////////////////////////////////////////////////////////////////
	/////////////////////////////////////////////////////////////////////////
	//Choice 2: Read in
	else{
		string filename;
		cout << "You picked choice 2!";
		cout << "Enter name of file with the homography matrices: ";
		cin >> filename;
		cout << "\nYou entered the filename " << filename << "\n";

		// Read in homography matrices
		ifstream readh;
		readh.open(filename);
		vector <double[3]> hs;
		int counter = 0;
		int i = 0;
		vector<Mat> thematrices;
		double dataforMat[3][3];
		while (readh >> dataforMat[0][0] >> dataforMat[0][1] >> dataforMat[0][2] >> dataforMat[1][0] >> dataforMat[1][1] >> dataforMat[1][2] >> dataforMat[2][0] >> dataforMat[2][1] >> dataforMat[2][2]){
			thematrices.push_back(Mat(3, 3, CV_64FC1, dataforMat));
			cout << thematrices[i];
			i++;
		}

		int numh = thematrices.size();
		vector<Mat> theimages;

		// Read in image files, named i.jpg for i = 1,2,3,.....
		for (int n = 0; n < numh; n++){
			theimages.push_back(imread("frame" + std::to_string((long long)n) + ".jpg"));
		}
		return 0;
	}
	return 0;
}