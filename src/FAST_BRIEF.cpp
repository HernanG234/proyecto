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


// ratio threshold on the nearest neighbor distances (as in SIFT paper, default should be 0.8)
static const float nn_ratio_threshold = 0.8f;

int matchHamming(const Mat& descriptors1,
                 const vector<Mat>& descriptors2,
                 vector< vector<DMatch> >& matches)
{
	 const int maxDist = descriptors1.cols<<3;
	 matches.resize(descriptors1.rows);
	 #pragma omp parallel for
	 for (int i=0; i<descriptors1.rows; ++i)
	 {
		 int minDist1 = maxDist, minDist2 = maxDist;
		 int minId1 = -1, minId2 = -1;
		 int minIdImg1 = -1, minIdImg2 = -1;
		 const long long* descr_i = descriptors1.ptr<long long>(i);
		 for (unsigned int imgIdx=0; imgIdx<descriptors2.size(); ++imgIdx)
		 {
			for (int j=0; j<descriptors2[imgIdx].rows; ++j)
			{
				  int currentDist = 0;
				  const long long* descr_j = descriptors2[imgIdx].ptr<long long>(j);
				  for (int d=0; d<descriptors1.cols/8; ++d)
					  {
						 currentDist += __builtin_popcountll(descr_i[d]^descr_j[d]);
					  }
			if (currentDist<minDist1)
			{
			    minDist2 = minDist1;
			    minId2 = minId1;
			    minIdImg2 = minIdImg1;
			    minDist1 = currentDist;
			    minId1 = j;
				minIdImg1 = imgIdx;
			}
			else if (currentDist>=minDist1 && currentDist<minDist2)
			{
				minDist2 = currentDist;
				minId2 = j;
				minIdImg2 = imgIdx;
			}
		}
	  }
	  #pragma omp critical
	  {
	   matches[i].push_back(DMatch(i, minId1, minIdImg1, (float) minDist1));
	   matches[i].push_back(DMatch(i, minId2, minIdImg2, (float) minDist2));
	  }
	 }
	 return 0;
}

int matchHamming(const Mat& descriptors1,
                 const Mat& descriptors2,
                 vector< vector<DMatch> >& matches)
{
	vector<Mat> temp;
	temp.push_back(descriptors2);
	return matchHamming(descriptors1, temp, matches);
}

void saveMatchedImages(Mat image1,
                       Mat image2,
                       const vector<KeyPoint>& keypoints1,
                       const vector<KeyPoint>& keypoints2,
                       const vector<DMatch>& homography_matches,
                       const Mat& H,
                       const string& outputImgFile)
{
//	Mat image1 = imread(img1Filename, CV_LOAD_IMAGE_GRAYSCALE);
//	Mat image2 = imread(img2Filename, CV_LOAD_IMAGE_GRAYSCALE);

	 // get the corners from the object image
	vector<Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( image1.cols, 0 );
	obj_corners[2] = cvPoint( image1.cols, image1.rows ); obj_corners[3] = cvPoint( 0, image1.rows );
	vector<Point2f> scene_corners(4);

	perspectiveTransform( obj_corners, scene_corners, H);

	 // draw matches
	Mat img_matches;
	drawMatches( image1, keypoints1, image2, keypoints2,
		      homography_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		      vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

       // draw lines between the corners (the mapped object in the scene - image )
   /* line( img_matches,
	scene_corners[0] + Point2f( image1.cols, 0), scene_corners[1] + Point2f( image1.cols, 0), Scalar(0, 255, 0), 4 );
    line( img_matches,
	scene_corners[1] + Point2f( image1.cols, 0), scene_corners[2] + Point2f( image1.cols, 0), Scalar( 0, 255, 0), 4 );
	line( img_matches,
	scene_corners[2] + Point2f( image1.cols, 0), scene_corners[3] + Point2f( image1.cols, 0), Scalar( 0, 255, 0), 4 );
	line( img_matches,
	scene_corners[3] + Point2f( image1.cols, 0), scene_corners[0] + Point2f( image1.cols, 0), Scalar( 0, 255, 0), 4 );
	*/
 // save (or show) detected matches
 imwrite(outputImgFile.c_str(), img_matches);
 namedWindow( "Matches", WINDOW_AUTOSIZE );// Create a window for display.
 imshow( "Matches", img_matches );
 #if VERBOSE
 fprintf(stdout, "[INFO] Matches saved to '%s'.\n", outputImgFile.c_str());
 #endif
}

int main(int argc, char** argv)
{
	double t1,t2,tdet;
	Mat src_1,src_2, descriptors_1, descriptors_2;
	vector<KeyPoint> keypoints_1, keypoints_2;
	src_1 = imread("images/000000.png", CV_LOAD_IMAGE_GRAYSCALE);
	src_2 = imread("images/000001.png", CV_LOAD_IMAGE_GRAYSCALE);
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

	////ESTE MATCHING ES MAS RAPIDO, PROBAR SI NO SE PUEDE HACER DE ESTA FORMA////
	//Matching
	/*BFMatcher matcher(NORM_HAMMING);
	std::vector< DMatch > matches;
	t1 = cv::getTickCount();
	matcher.match(descriptors_1, descriptors_2, matches );
	t2 = cv::getTickCount();
	tdet = 1000.0*(t2-t1) / cv::getTickFrequency();
	cout<<"Matching: "<< descriptors_1.rows<<" descriptores (imagen 1), contra "<< descriptors_2.rows<<" descriptores (imagen 2)"<<endl;
	cout<<"Tiempo de matching: "<<tdet<<" ms"<<endl; 

	//-- Draw matches
	Mat img_matches;
	drawMatches( src_1, keypoints_1, src_2, keypoints_2, matches, img_matches);
	imshow("matches",img_matches);*/
	////////////////////////////////////////////////////////////////////////////

	vector<vector<DMatch> > knnMatches;
	matchHamming(descriptors_1, descriptors_2, knnMatches);
	t1 = cv::getTickCount();
	matchHamming(descriptors_1, descriptors_2, knnMatches);
	t2 = cv::getTickCount();
	tdet = 1000.0*(t2-t1) / cv::getTickFrequency();
	cout<<"Matching: "<< descriptors_1.rows <<" descriptores (imagen 1), contra "<< descriptors_2.rows<<" descriptores (imagen 2)"<<endl;
	cout<<"Tiempo de matching: "<<tdet<<" ms"<<endl;

	cout<<"Matches: "<<knnMatches.size()<<endl;
	vector< DMatch > good_matches;
	for(unsigned int i = 0; i < knnMatches.size(); i++ )
		if (knnMatches[i][0].distance<nn_ratio_threshold *knnMatches[i][1].distance)
			good_matches.push_back( knnMatches[i][0]);

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
	/*
	fprintf(stdout, "Numero de inliers(matches post-homography): %d (%3.2f%% of the RANSAC input).\n",
		(int) homography_matches.size(), 100.f * (float) homography_matches.size() / (float) good_matches.size()); */

	// display matched images
	const string outputImgFile = "matches.png";
	saveMatchedImages(src_1, src_2, keypoints_1, keypoints_2, homography_matches, H, outputImgFile);

	waitKey(0);
	return 0;
}

