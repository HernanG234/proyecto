// standard headers
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string>

// OpenCv headers
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
//#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;
using namespace xfeatures2d;

static const float nn_ratio_threshold = 0.8f;

int main(int argc, char** argv)
{
	double t1,t2,tdet;
	Mat src_1,src_2, descriptors_1, descriptors_2;
	vector<KeyPoint> keypoints_1, keypoints_2;
	src_1 = imread("src/images/000000.png", CV_LOAD_IMAGE_GRAYSCALE);
	src_2 = imread("src/images/000001.png", CV_LOAD_IMAGE_GRAYSCALE);
	//src_1 = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	//src_2 = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);

	Ptr<FastFeatureDetector> detector_1=FastFeatureDetector::create(106);
	Ptr<FastFeatureDetector> detector_2=FastFeatureDetector::create(106);
	Ptr<BriefDescriptorExtractor> featureExtractor_1 = BriefDescriptorExtractor::create();
	Ptr<BriefDescriptorExtractor> featureExtractor_2 = BriefDescriptorExtractor::create();
	detector_1->detect(src_1,keypoints_1,Mat());

	//Deteccion imagen 1
	t1 = cv::getTickCount();
	detector_1->detect(src_1,keypoints_1,Mat());
	t2 = cv::getTickCount();
	tdet = 1000.0*(t2-t1) / cv::getTickFrequency();
	//cout<<"Imagen 1: "<<endl;
	cout<<"Cantidad de Keypoints: "<<keypoints_1.size()<<endl;
	cout<<"Tiempo deteccion (FAST): "<<tdet<<" ms"<<endl;

	//Deteccion imagen 2
	//t1 = cv::getTickCount();
	detector_2->detect(src_2,keypoints_2,Mat());
	/*t2 = cv::getTickCount();
	tdet = 1000.0*(t2-t1) / cv::getTickFrequency();
	cout<<"Imagen 2: "<<endl;
	cout<<"Cantidad de Keypoints : "<<keypoints_2.size()<<endl;
	cout<<"Tiempo deteccion (FAST): "<<tdet<<" ms"<<endl; */
	//KeyPointsFilter::retainBest(keypoints, 500);

	drawKeypoints(src_1, keypoints_1, src_1);
	drawKeypoints(src_2, keypoints_2, src_2);
	imshow("keypoints",src_1);
	imshow("keypoints_2",src_2);

	//Descripcion imagen 1
	t1 = cv::getTickCount();
	featureExtractor_1->compute(src_1, keypoints_1, descriptors_1);
	t2 = cv::getTickCount();
	tdet = 1000.0*(t2-t1) / cv::getTickFrequency();
	//cout<<"Imagen 1: "<<endl;
	cout<<"Tiempo de descripcion (BRIEF): "<<tdet<<" ms"<<endl;

	//Descripcion imagen 2
	//t1 = cv::getTickCount(); 
	featureExtractor_2->compute(src_2, keypoints_2, descriptors_2);
	/*t2 = cv::getTickCount();
	tdet = 1000.0*(t2-t1) / cv::getTickFrequency();
	cout<<"Imagen 2: "<<endl;
	cout<<"Tiempo de descripcion (BRIEF): "<<tdet<<" ms"<<endl;*/

	cout<<descriptors_1.size()<<endl;
	cout<<descriptors_2.size()<<endl;

	//Matching
	BFMatcher matcher(NORM_HAMMING);
	vector<vector<DMatch> > matches;
	t1 = cv::getTickCount();
	matcher.knnMatch(descriptors_1, descriptors_2, matches, 2);
	t2 = cv::getTickCount();
	tdet = 1000.0*(t2-t1) / cv::getTickFrequency();
	cout<<"Matching: "<< descriptors_1.rows<<" descriptores (imagen 1), contra "<< descriptors_2.rows<<" descriptores (imagen 2)"<<endl;
	cout << "Tiempo de matching: " << tdet << " ms" << endl;

	cout << "Matches: " << matches.size()<<endl;
	vector<DMatch> good_matches;
	for(unsigned int i = 0; i < matches.size(); i++ )
		if (matches[i][0].distance<nn_ratio_threshold *matches[i][1].distance)
			good_matches.push_back(matches[i][0]);

	cout<<"Good matches: "<<good_matches.size()<<endl;

	vector<Point2f> match_left, match_right;
	vector<float> distances;

	for(unsigned int i = 0; i < good_matches.size(); i++ )
	{
		match_left.push_back( keypoints_1[ good_matches[i].queryIdx ].pt );
		match_right.push_back( keypoints_2[ good_matches[i].trainIdx ].pt );
		distances.push_back(good_matches[i].distance);
	}

	Mat correctMatches;
	Mat H = findHomography( match_left, match_right, CV_RANSAC, 3, correctMatches );

	// check if the homography matrix is good enough
	float detMin = 1e-3f;
	if (abs(determinant(H)) < detMin)
		cout<<"Mala Homografia (|det(H)| < "<<detMin<<endl;

	vector<DMatch> homography_matches;
	for (unsigned int i=0; i < good_matches.size(); ++i)
	{
		if (*correctMatches.ptr<uchar>(i))
			homography_matches.push_back(good_matches[i]);
	}

	cout<<"Matches correctos (inliers): "<<	homography_matches.size() <<
		" ("<<100.f * (float) homography_matches.size() / (float) good_matches.size()<<"%)"<<endl;
	// Draw matches
	Mat img_matches;
	drawMatches( src_1, keypoints_1, src_2, keypoints_2, homography_matches, img_matches);
	imshow("matches",img_matches);
	// Save Image
	imwrite("matches.png", img_matches);

	waitKey(0);
	return 0;
}

