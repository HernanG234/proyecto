// standard headers
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>


// OpenCv headers
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
//#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <bitset>
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>


#include "ldb.h"
#include "LATCHK.h"

#define VC_EXTRALEAN
#define WIN32_LEAN_AND_MEAN
#include "baft.h"

using namespace std;
using namespace cv;
using namespace xfeatures2d;

static const float nn_ratio_threshold = 0.8f;

double calc_detection(Ptr<FeatureDetector> detector, Mat &img, vector<KeyPoint> &keypoints, bool gettime)
{
	double t1,t2,tdet=0;

	if (gettime)
	{
		for(int i=0; i<30;i++){
			t1 = cv::getTickCount();
			detector->detect(img, keypoints);
			t2 = cv::getTickCount();
			tdet += 1000.0*(t2-t1) / cv::getTickFrequency();
		}

		tdet/=30;
		cout <<"Cantidad de Keypoints: "<< keypoints.size() << endl;
		cout << "Tiempo deteccion: " << tdet << " ms" << endl;
	}
	else
		detector->detect(img, keypoints);

	if(gettime) return tdet;
	else return 0;
}

double calc_description(Ptr<Feature2D> extractor, Mat &img, vector<KeyPoint> &keypoints, Mat &descriptors, bool gettime)
{
	double t1,t2,tdesc=0;

	if (gettime) {
		extractor->compute(img, keypoints, descriptors);
		for(int i=0; i<30;i++){
			t1 = cv::getTickCount();
			extractor->compute(img, keypoints, descriptors);
			t2 = cv::getTickCount();
			tdesc += 1000.0*(t2-t1) / cv::getTickFrequency();
		}
		tdesc/=30;
		cout<<"Tiempo de descripcion: "<<tdesc<<" ms"<<endl;
	}
	else
		extractor->compute(img, keypoints, descriptors);

	cout<<"Descriptor size: "<<descriptors.size()<<endl;

	if(gettime) return tdesc;
	else return 0;
}

int main(int argc, char** argv)
{
	constexpr bool multithread = true;
	double t1,t2,tdet,tdesc=0,tmatch=0;
	Mat src_1,src_2, descriptors_1, descriptors_2;

	vector<KeyPoint> keypoints_1, keypoints_2;
	int kpts;
	src_1 = imread("src/images/000000.png", CV_LOAD_IMAGE_GRAYSCALE);
	src_2 = imread("src/images/000001.png", CV_LOAD_IMAGE_GRAYSCALE);
	//src_1 = imread("src/images/000106.png", CV_LOAD_IMAGE_GRAYSCALE);
	//src_2 = imread("src/images/000107.png", CV_LOAD_IMAGE_GRAYSCALE);
	//src_1 = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	//src_2 = imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);

	//Ayuda
	if(argc!=3){
		cout<<"Modo de uso: ./ejecutable <detector> <descriptor>"<<endl;
		cout<<"Detector: FAST, ORB o GFTT"<<endl;
		cout<<"Descriptor: BRIEF, BRISK, FREAK ,(los que probemos)"<<endl;
		return 0;
	}

	//Si el detector es FAST
	if( !strcmp("FAST", argv[1] )){
		/* void FAST(InputArray image, vector<KeyPoint>& keypoints, int threshold, bool nonmaxSuppression=true )*/

		Ptr<FastFeatureDetector> detector=FastFeatureDetector::create(106);
		//Ptr<FastFeatureDetector> detector_2=FastFeatureDetector::create(106);
		detector->detect(src_1,keypoints_1,Mat());
		tdet=calc_detection(detector, src_1, keypoints_1, true);
		//cout<< tdet<<endl;
		calc_detection(detector, src_2, keypoints_2, false);
		//cout<< tdet<<endl;
	}

	//Si el detector es ORB
	else if( !strcmp("ORB", argv[1] )){
		/*Parametros por defecto:
		ORB(int nfeatures=500, float scaleFactor=1.2f,
		int nlevels=8, int edgeThreshold=31, int firstLevel=0,
		int WTA_K=2, int scoreType=ORB::HARRIS_SCORE, int patchSize=31)*/

		Ptr<FeatureDetector> detector = ORB::create(1000);
		detector->detect(src_1, keypoints_1);
		calc_detection(detector, src_1, keypoints_1, true);
		calc_detection(detector, src_2, keypoints_2, false);
	}

	//Si el detector es BRISK
	else if( !strcmp("BRISK", argv[1] )){
		/*Parametros por defecto:
		BUSCAR PARAMETROS POR DEFECTO BRISK
		*/

		Ptr<FeatureDetector> detector = BRISK::create(130);
		detector->detect(src_1, keypoints_1);
		tdet=calc_detection(detector, src_1, keypoints_1, true);
		calc_detection(detector, src_2, keypoints_2, false);
	}

	//Si el detector es AGAST
	else if( !strcmp("AGAST", argv[1] )){
		/*Parametros por defecto:

		  cv::AgastFeatureDetector::create(int threshold = 10, bool nonmaxSuppression = true,
		  int type = AgastFeatureDetector::OAST_9_16)
		 */
		Ptr<FeatureDetector> detector = AgastFeatureDetector::create(130, false);
		detector->detect(src_1, keypoints_1);
		tdet=calc_detection(detector, src_1, keypoints_1, true);
		calc_detection(detector, src_2, keypoints_2, false);
	}

	//Si el detector es GFTT
	else if( !strcmp("GFTT", argv[1] )){
		/*Parametros por defecto:
		  static Ptr< GFTTDetector > 	create (int maxCorners=1000,
		  double qualityLevel=0.01,
		  double minDistance=1, int blockSize=3,
		  bool useHarrisDetector=false, double k=0.04)*/

		//tarda lo mismo en detectar 500 o 1000 puntos
		Ptr<FeatureDetector> detector = GFTTDetector::create(500);
		detector->detect(src_1, keypoints_1);
		tdet=calc_detection(detector, src_1, keypoints_1, true);
		calc_detection(detector, src_2, keypoints_2, false);
	}

		//Si el detector es BAFT
	else if( !strcmp("BAFT", argv[1] )){
		/*Parametros por defecto:
		ORB(int nfeatures=500, float scaleFactor=1.2f,
		int nlevels=8, int edgeThreshold=31, int firstLevel=0,
		int WTA_K=2, int scoreType=ORB::HARRIS_SCORE, int patchSize=31)*/

		Ptr<Feature2D> detector = BAFT::create(500,64);
		detector->detect(src_1, keypoints_1);
		tdet=calc_detection(detector, src_1, keypoints_1, true);
		calc_detection(detector, src_2, keypoints_2, false);
	}

	else{
		cout<<argv[1]<<" no es un nombre de detector valido"<<endl;
		cout<<"Detectores: FAST,ORB,GFTT,BAFT"<<endl;
		return 0;
	}
	kpts=keypoints_1.size();

	kpts=keypoints_1.size();

	//Descriptores
	//Si el descriptor es BRIEF

	if( !strcmp("BRIEF", argv[2] )){
		/*Parametros por defecto:
		  static Ptr< BriefDescriptorExtractor > 	create (int bytes=32, bool use_orientation=false)*/

		Ptr<BriefDescriptorExtractor> featureExtractor = BriefDescriptorExtractor::create();
		//Ptr<BriefDescriptorExtractor> featureExtractor_2 = BriefDescriptorExtractor::create();

		tdesc=calc_description(featureExtractor, src_1, keypoints_1, descriptors_1, true);
		calc_description(featureExtractor, src_2, keypoints_2, descriptors_2, false);
	}

	//Si el descriptor es BRISK
	else if( !strcmp("BRISK", argv[2] )){
		/*Parametros por defecto:
		  BRISK::BRISK(int thresh=30, int octaves=3, float patternScale=1.0f)*/
		Ptr<Feature2D> featureExtractor = BRISK::create();

		tdesc=calc_description(featureExtractor, src_1, keypoints_1, descriptors_1, true);
		calc_description(featureExtractor, src_2, keypoints_2, descriptors_2, false);
	}

	//Si el descriptor es FREAK
	else if( !strcmp("FREAK", argv[2] )){

		/*static Ptr< FREAK > 	create (bool orientationNormalized=true, bool scaleNormalized=true,
		  float patternScale=22.0f, int 	nOctaves=4, const std::vector< int > &selectedPairs=std::vector< int >())*/

		Ptr<Feature2D> featureExtractor = FREAK::create();
		//extractor->compute(src_1, keypoints_1, descriptors_1);
		tdesc=calc_description(featureExtractor, src_1, keypoints_1, descriptors_1, true);
		calc_description(featureExtractor, src_2, keypoints_2, descriptors_2, false);
	}

	//Si el descriptor es ORB
	else if( !strcmp("ORB", argv[2] )){
		/*static Ptr< ORB >	create (int nfeatures=500, float scaleFactor=1.2f, int nlevels=8, int edgeThreshold=31, int
		  firstLevel=0, int WTA_K=2, int scoreType=ORB::HARRIS_SCORE, int patchSize=31, int
		  fastThreshold=20)*/
		Ptr<Feature2D> featureExtractor = ORB::create(500);

		tdesc=calc_description(featureExtractor, src_1, keypoints_1, descriptors_1, true);
		calc_description(featureExtractor, src_2, keypoints_2, descriptors_2, false);
	}

	//Si el descriptor es LATCH
	else if( !strcmp("LATCH", argv[2] )){
		/*static Ptr<LATCH> cv::xfeatures2d::LATCH::create(int bytes = 32,
		 *bool rotationInvariance = true, int half_ssd_size = 3, double sigma = 2.0)		
		 */
		Ptr<Feature2D> featureExtractor = LATCH::create(64);

		calc_description(featureExtractor, src_1, keypoints_1, descriptors_1, true);
		calc_description(featureExtractor, src_2, keypoints_2, descriptors_2, false);
		cout << "Descriptor OpenCV LATCH size: " << descriptors_1.size() << endl;
		cout << descriptors_2.row(0) << endl;
	}

	// ------------- LATCHK ------------
	else if( !strcmp("LATCHK", argv[2] )){
		uint64_t* desc_1 = new uint64_t[8 * keypoints_1.size()];
		std::vector<KeyPointK> kps1;
		for (auto&& kp : keypoints_1) kps1.emplace_back(kp.pt.x, kp.pt.y, kp.size, kp.angle * 3.14159265f / 180.0f);
		LATCHK<multithread>(src_1.data, src_1.cols, src_1.rows, static_cast<int>(src_1.step), kps1, desc_1);
		t1 = cv::getTickCount();
		for (int i=0; i<30; i++)
			LATCHK<multithread>(src_1.data, src_1.cols, src_1.rows, static_cast<int>(src_1.step), kps1, desc_1);
		for (size_t i=0; i < 8 * kps1.size(); ++i)
			desc_1[i] =  __builtin_bswap64 (desc_1[i]);
		descriptors_1 = Mat(keypoints_1.size(), 64, CV_8U, desc_1, 64);
		t2 = cv::getTickCount();
		tdet = 1000.0*(t2-t1) / cv::getTickFrequency() / 30;
		cout<<"Tiempo de descripcion: "<<tdet<<" ms"<<endl;
		// --------------------------------

		// ------------- LATCHK ------------
		uint64_t* desc_2 = new uint64_t[8 * keypoints_2.size()];
		std::vector<KeyPointK> kps2;
		for (auto&& kp : keypoints_2) kps2.emplace_back(kp.pt.x, kp.pt.y, kp.size, kp.angle * 3.14159265f / 180.0f);
		LATCHK<multithread>(src_2.data, src_2.cols, src_2.rows, static_cast<int>(src_2.step), kps2, desc_2);
		LATCHK<multithread>(src_2.data, src_2.cols, src_2.rows, static_cast<int>(src_2.step), kps2, desc_2);
		for (size_t i=0; i < 8 * kps2.size(); ++i)
			desc_2[i] =  __builtin_bswap64 (desc_2[i]);
		descriptors_2 = Mat(keypoints_2.size(), 64, CV_8U, desc_2, 64);
		// -------------------------------- 
	}

	//Si el descriptor es LDB
	else if( !strcmp("LDB", argv[2] )){
		//LDB(int _bytes = 32, int _nlevels = 3, int _patchSize = 60);
		//Feature2D featureExtractor = LdbDescriptorExtractor::create();

		LDB featureExtractor(32);
		t1 = cv::getTickCount();
		featureExtractor.compute(src_1, keypoints_1, descriptors_1, 0);
		for(int i=0;i<30;i++){
			t1 = cv::getTickCount();
			featureExtractor.compute(src_1, keypoints_1, descriptors_1, 0);
			t2 = cv::getTickCount();
			tdesc += 1000.0*(t2-t1) / cv::getTickFrequency();
		}
		tdesc/=30;
		cout <<"Cantidad de Keypoints: "<< keypoints_1.size() << endl;
		cout << "Tiempo descripcion: " << tdet << " ms" << endl;
		featureExtractor.compute(src_2, keypoints_2, descriptors_2, 0);
		cout<<"Descriptor size: "<<descriptors_1.size()<<endl;
		//calc_description(&featureExtractor, src_1, keypoints_1, descriptors_1, true);
		//calc_description(&featureExtractor, src_2, keypoints_2, descriptors_2, false);
	}

	//si el descriptor es BAFT
	else if(!strcmp("BAFT", argv[2])){
		Ptr<Feature2D> featureExtractor = BAFT::create(500,64);
		tdesc=calc_description(featureExtractor, src_1, keypoints_1, descriptors_1, true);
		calc_description(featureExtractor, src_2, keypoints_2, descriptors_2, false);
	}

	else{
		cout<<argv[2]<<" no es un nombre de descriptor valido"<<endl;
		cout<<"Descriptores: BRIEF,BRISK,FREAK,LATCH,LDB,BAFT,(...)"<<endl;
		return 0;
	}

	//Dibujar kpts en las dos imagenes
	drawKeypoints(src_1, keypoints_1, src_1);
	drawKeypoints(src_2, keypoints_2, src_2);
	imshow("keypoints",src_1);
	imshow("keypoints_2",src_2);

	//Matching
	BFMatcher matcher(NORM_HAMMING);
	vector<vector<DMatch> > matches,matches_1;
	for(int i=0;i<50;i++){
		t1 = cv::getTickCount();
		matcher.knnMatch(descriptors_1, descriptors_2, matches_1, 2);
		t2 = cv::getTickCount();
		tmatch += 1000.0*(t2-t1) / cv::getTickFrequency();
	}
	tmatch/= 50;
	cout<<"Matching: "<< descriptors_1.rows<<" descriptores (imagen 1), contra "<< descriptors_2.rows<<" descriptores (imagen 2)"<<endl;
	cout << "Tiempo de matching: " << tmatch << " ms" << endl;
	matcher.knnMatch(descriptors_1, descriptors_2, matches, 2);
	//cout << "Matches: " << matches.size()<<endl;
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
	drawMatches( src_1, keypoints_1, src_2, keypoints_2, good_matches, img_matches);
	imshow("matches",img_matches);
	// Save Image
	imwrite("matches.png", img_matches);

	//Archivo para guardar resultados
	ofstream file("Resultados.txt", ios_base::app);
	//file.open()
	file<<argv[1]<<" + "<<argv[2]<<":"<<endl;
	//file <<"Cantidad de Keypoints: "<< keypoints.size() << endl;
	file <<"Cantidad de Keypoints: "<< kpts << endl;
	file<<"Descriptor size: "<<descriptors_1.size()<<endl;
	file<<"Tiempo deteccion: "<<tdet<<" ms "<<endl;
	file<<"Tiempo descripcion: "<<tdesc<<" ms "<<endl;
	file<<"Tiempo descripcion por keypoint: "<<tdesc*1000/descriptors_1.rows<<" us "<<endl;
	file<<"Tiempo Match: "<<tmatch<<" ms "<<endl;
	file<<"Tiempo total: "<<tdet+tdesc+tmatch<<" ms "<<endl;
	file<<"Good matches: "<<good_matches.size()<<endl<<endl;
	//file<<"Matches correctos (inliers): "<<	homography_matches.size() <<
		//" ("<<100.f * (float) homography_matches.size() / (float) good_matches.size()<<"%)"<<endl<<endl;
	waitKey(0);
	return 0;
}
