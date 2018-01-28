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

	//Ayuda
	if(argc!=3){
		cout<<"Modo de uso: ./ejecutable <detector> <descriptor>"<<endl;
		cout<<"Detector: FAST,ORB o GFTT"<<endl;
		cout<<"Descriptor:BRIEF, BRISK, FREAK ,(los que probemos)"<<endl;
		return 0;
	}

	//Si el detector es FAST
	if( !strcmp("FAST", argv[1] )){	

		/* void FAST(InputArray image, vector<KeyPoint>& keypoints, int threshold, bool nonmaxSuppression=true )*/
		
		Ptr<FastFeatureDetector> detector=FastFeatureDetector::create(106);
		//Ptr<FastFeatureDetector> detector_2=FastFeatureDetector::create(106);
		detector->detect(src_1,keypoints_1,Mat());

		//Deteccion imagen 1
		t1 = cv::getTickCount();
		detector->detect(src_1,keypoints_1,Mat());
		t2 = cv::getTickCount();
		tdet = 1000.0*(t2-t1) / cv::getTickFrequency();
		//cout<<"Imagen 1: "<<endl;
		cout<<"Cantidad de Keypoints: "<<keypoints_1.size()<<endl;
		cout<<"Tiempo deteccion (FAST): "<<tdet<<" ms"<<endl;

		//Deteccion imagen 2
		//t1 = cv::getTickCount();
		detector->detect(src_2,keypoints_2,Mat());
		/*t2 = cv::getTickCount();
		tdet = 1000.0*(t2-t1) / cv::getTickFrequency();
		cout<<"Imagen 2: "<<endl;
		cout<<"Cantidad de Keypoints : "<<keypoints_2.size()<<endl;
		cout<<"Tiempo deteccion (FAST): "<<tdet<<" ms"<<endl; */
		//KeyPointsFilter::retainBest(keypoints, 500);
	}

	//Si el detector es ORB

	else if( !strcmp("ORB", argv[1] )){

		/*Parametros por defecto:
		ORB(int nfeatures=500, float scaleFactor=1.2f, int nlevels=8, int edgeThreshold=31, int firstLevel=0, int WTA_K=2, int 				scoreType=ORB::HARRIS_SCORE, int patchSize=31)*/

		Ptr<FeatureDetector> detector = ORB::create();
		detector->detect(src_1, keypoints_1);

		//Deteccion imagen 1
		t1 = cv::getTickCount();
		detector->detect(src_1, keypoints_1);
		t2 = cv::getTickCount();
		tdet = 1000.0*(t2-t1) / cv::getTickFrequency();
		cout<<"Cantidad de Keypoints : "<<keypoints_1.size()<<endl;
		cout<<"Tiempo deteccion (ORB): "<<tdet<<" ms"<<endl; 

		//Deteccion imagen 2
		//t1 = cv::getTickCount();
		detector->detect(src_2, keypoints_2);
		/*t2 = cv::getTickCount();
		tdet = 1000.0*(t2-t1) / cv::getTickFrequency();
		cout<<"Imagen 2: "<<endl;
		cout<<"Cantidad de Keypoints : "<<keypoints_2.size()<<endl;
		cout<<"Tiempo deteccion (ORB): "<<tdet<<" ms"<<endl; */
		//KeyPointsFilter::retainBest(keypoints, 500);

	}


	//Si el detector es GFTT

	else if( !strcmp("GFTT", argv[1] )){

		/*Parametros por defecto:
		static Ptr< GFTTDetector > 	create (int maxCorners=1000, double qualityLevel=0.01, double minDistance=1, int 				blockSize=3, bool useHarrisDetector=false, double k=0.04)*/
		
		//tarda lo mismo en detectar 500 o 1000 puntos
		Ptr<FeatureDetector> detector = GFTTDetector::create(500);
		detector->detect(src_1, keypoints_1);

		//Deteccion imagen 1
		t1 = cv::getTickCount();
		detector->detect(src_1, keypoints_1);
		t2 = cv::getTickCount();
		tdet = 1000.0*(t2-t1) / cv::getTickFrequency();
		cout<<"Cantidad de Keypoints : "<<keypoints_1.size()<<endl;
		cout<<"Tiempo deteccion (GFTT): "<<tdet<<" ms"<<endl;

		//Deteccion imagen 2
		//t1 = cv::getTickCount();
		detector->detect(src_2, keypoints_2);
		/*t2 = cv::getTickCount();
		tdet = 1000.0*(t2-t1) / cv::getTickFrequency();
		cout<<"Imagen 2: "<<endl;
		cout<<"Cantidad de Keypoints : "<<keypoints_2.size()<<endl;
		cout<<"Tiempo deteccion (GFTT): "<<tdet<<" ms"<<endl; */
		//KeyPointsFilter::retainBest(keypoints, 500);
	}

	else{
		cout<<argv[1]<<" no es un nombre de detector valido"<<endl;
		cout<<"Detectores: FAST,ORB o GFTT"<<endl;
		return 0;
	}

	//Descriptores

	//Si el descriptor es BRIEF

	
	if( !strcmp("BRIEF", argv[2] )){

		/*Parametros por defecto:
		static Ptr< BriefDescriptorExtractor > 	create (int bytes=32, bool use_orientation=false)*/
		
		Ptr<BriefDescriptorExtractor> featureExtractor = BriefDescriptorExtractor::create();
		//Ptr<BriefDescriptorExtractor> featureExtractor_2 = BriefDescriptorExtractor::create();
		
		//Descripcion imagen 1
		t1 = cv::getTickCount();
		featureExtractor->compute(src_1, keypoints_1, descriptors_1);
		t2 = cv::getTickCount();
		tdet = 1000.0*(t2-t1) / cv::getTickFrequency();
		//cout<<"Imagen 1: "<<endl;
		cout<<"Tiempo de descripcion (BRIEF): "<<tdet<<" ms"<<endl;

		//Descripcion imagen 2
		//t1 = cv::getTickCount(); 
		featureExtractor->compute(src_2, keypoints_2, descriptors_2);
		/*t2 = cv::getTickCount();
		tdet = 1000.0*(t2-t1) / cv::getTickFrequency();
		cout<<"Imagen 2: "<<endl;
		cout<<"Tiempo de descripcion (BRIEF): "<<tdet<<" ms"<<endl;*/

		cout<<descriptors_1.size()<<endl;
		cout<<descriptors_2.size()<<endl;

	}
	
	//Si el descriptor es BRISK

	else if( !strcmp("BRISK", argv[2] )){

		/*Parametros por defecto:
		BRISK::BRISK(int thresh=30, int octaves=3, float patternScale=1.0f)*/
		
		Ptr<Feature2D> featureExtractor = BRISK::create();

		//Descripcion imagen 1
		featureExtractor->compute(src_1, keypoints_1, descriptors_1);
		t1 = cv::getTickCount();
		featureExtractor->compute(src_1, keypoints_1, descriptors_1);
		t2 = cv::getTickCount();
		tdet = 1000.0*(t2-t1) / cv::getTickFrequency();
		//cout<<"Imagen 1: "<<endl;
		cout<<"Tiempo de descripcion (BRISK): "<<tdet<<" ms"<<endl;


		//Descripcion imagen 2
		//t1 = cv::getTickCount(); 
		featureExtractor->compute(src_2, keypoints_2, descriptors_2);
		/*t2 = cv::getTickCount();
		tdet = 1000.0*(t2-t1) / cv::getTickFrequency();
		cout<<"Imagen 2: "<<endl;
		cout<<"Tiempo de descripcion (BRISK): "<<tdet<<" ms"<<endl;*/

		cout<<descriptors_1.size()<<endl;
		cout<<descriptors_2.size()<<endl;
	}

	//Si el descriptor es FREAK

	else if( !strcmp("FREAK", argv[2] )){
		
		/*static Ptr< FREAK > 	create (bool orientationNormalized=true, bool scaleNormalized=true,
 		float patternScale=22.0f, int 	nOctaves=4, const std::vector< int > &selectedPairs=std::vector< int >())*/

		Ptr<Feature2D> featureExtractor = FREAK::create(); 

		//Descripcion imagen 1
		featureExtractor->compute(src_1, keypoints_1, descriptors_1);
		t1 = cv::getTickCount();
		featureExtractor->compute(src_1, keypoints_1, descriptors_1);
		t2 = cv::getTickCount();
		tdet = 1000.0*(t2-t1) / cv::getTickFrequency();
		//cout<<"Imagen 1: "<<endl;
		cout<<"Tiempo de descripcion (FREAK): "<<tdet<<" ms"<<endl;


		//Descripcion imagen 2
		//t1 = cv::getTickCount(); 
		featureExtractor->compute(src_2, keypoints_2, descriptors_2);
		/*t2 = cv::getTickCount();
		tdet = 1000.0*(t2-t1) / cv::getTickFrequency();
		cout<<"Imagen 2: "<<endl;
		cout<<"Tiempo de descripcion (FREAK): "<<tdet<<" ms"<<endl;*/

		cout<<descriptors_1.size()<<endl;
		cout<<descriptors_2.size()<<endl;
		
	}

	else{
		cout<<argv[2]<<" no es un nombre de descriptor valido"<<endl;
		cout<<"Descriptores: BRIEF,BRISK,FREAK,(...)"<<endl;
		return 0;
	}


	//Dibujar kpts en las dos imagenes

	drawKeypoints(src_1, keypoints_1, src_1);
	drawKeypoints(src_2, keypoints_2, src_2);
	imshow("keypoints",src_1);
	imshow("keypoints_2",src_2);


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

