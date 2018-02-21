/*
    This file is part of LOCKY.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    
    Author: Jaime Lomeli-R.
*/


#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <locky.h>
#include <assert.h>

//---------------------------------------------------------------------------------------------------------------------
//-----------------------------LOCKYFeatureDetector_Impl---------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------

class LOCKYFeatureDetector_Impl : public locky::LOCKYFeatureDetector {

	public:
		LOCKYFeatureDetector_Impl(int iter = 100000, int maxExp = 7, int minExp = 3, int thresh = 30, bool bcts = true);
		~LOCKYFeatureDetector_Impl(void);

		void getAccumMat(cv::Mat& toReturn);

		// Detection methods
		void detect(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints);

	private:
		int	 iter;
		int	 maxExp;
		int	 minExp;
		int	 thresh;
		bool bcts;
		cv::Mat accumMat;

		void BCT(const cv::Mat& image);
		void BCTS(const cv::Mat& image);
		void extractLOCKY(std::vector<cv::KeyPoint>& keypoints);

};


// Constructor
LOCKYFeatureDetector_Impl::LOCKYFeatureDetector_Impl(int iter, int maxExp, int minExp, int thresh, bool bcts) {

	this->iter	 = iter;
	this->maxExp = maxExp;
	this->minExp = minExp;
	this->thresh = thresh;
	this->bcts	 = bcts;

	assert(iter > 0);
	assert(maxExp >= minExp);
	assert((thresh > 0) && (thresh < 255));

	accumMat = cv::Mat();
}


void LOCKYFeatureDetector_Impl::getAccumMat(cv::Mat& toReturn) {

	toReturn = accumMat.clone();
}


void LOCKYFeatureDetector_Impl::detect(const cv::Mat& image,
					std::vector<cv::KeyPoint>& keypoints) {

	if (bcts)
		BCTS(image);
	else
		BCT(image);
	
	extractLOCKY(keypoints);
}


void LOCKYFeatureDetector_Impl::BCT(const cv::Mat& image) {

	cv::Mat grayImage;

	assert(!image.empty());

    // Convert the image to gray scale 
    switch (image.type()) {

		case CV_32FC1:
			image.convertTo(grayImage, CV_8UC1, 255.0);
		break;

    	case CV_8UC3:
        	cvtColor(image, grayImage, CV_BGR2GRAY);
			grayImage.convertTo(grayImage, CV_8UC1);
        	break;

    	case CV_32FC3:
        	cvtColor(image, grayImage, CV_BGR2GRAY);
			grayImage.convertTo(grayImage, CV_8UC1, 255.0);
        	break;

    	default:
        	grayImage = image.clone();
        	break;
    }

	assert(grayImage.type() == CV_8UC1);
	assert(grayImage.isContinuous());

	
	accumMat = cv::Mat::zeros(grayImage.size(), CV_8UC1);

	// Init vars
	unsigned int  aux[9], weights[4], max;
	unsigned char idx;
	unsigned int  h, w, x, y;

	srand(time(0));

	// Create the integral Image
	cv::Mat integImage(grayImage.rows+1, grayImage.cols+1, CV_32FC1);
	cv::integral(grayImage, integImage);


	// Voting
	for (unsigned int it = 0; it < iter; it++) {


	    // Step 1: create rectangles
		h = 2;
		for(int i=2; i<=(minExp + rand() % (maxExp-minExp+1)); i++) h = h * 2;
		w = 2;
		for(int i=2; i<=(minExp + rand() % (maxExp-minExp+1)); i++) w = w * 2;
		x = rand() % (integImage.cols-w-1);
		y = rand() % (integImage.rows-h-1);
	
	    
        // Step 2: reduce rectangles
		while (h > 2 && w > 2) {
			h = h/2;
			w = w/2;
		
			aux[0] = integImage.at<int>(y,     x);
			aux[1] = integImage.at<int>(y,     x+w);
			aux[2] = integImage.at<int>(y,     x+2*w);
			aux[3] = integImage.at<int>(y+h,   x);
			aux[4] = integImage.at<int>(y+h,   x+w);
			aux[5] = integImage.at<int>(y+h,   x+2*w);
			aux[6] = integImage.at<int>(y+2*h, x);
			aux[7] = integImage.at<int>(y+2*h, x+w);
			aux[8] = integImage.at<int>(y+2*h, x+2*w);

			weights[0] = aux[0] + aux[4] - aux[1] - aux[3];
			weights[1] = aux[1] + aux[5] - aux[2] - aux[4];
			weights[2] = aux[3] + aux[7] - aux[4] - aux[6];
			weights[3] = aux[4] + aux[8] - aux[5] - aux[7];

			max = 0;
			idx = 0;
		
			for (unsigned char i = 0; i < 4; i++) {
				if (weights[i] > max){
					max = weights[i];
					idx = i;
				}
			}
		
			switch (idx) {
				case 0:
					break;
				case 1:
					x = x + w;
					break;
				case 2:
					y = y + h;
					break;
				case 3:
					x = x + w;
					y = y + h;
					break;
				
				default:
					break;
			}
		}

		// Step 3: vote pixel in coord
		unsigned int coord[2] = {y+h/2,x+w/2};
		accumMat.ptr<unsigned char>(coord[0])[coord[1]] += 1;
	}

	// Blur the transform image and normalize it
	cv::GaussianBlur(accumMat, accumMat, cv::Size(5,5), 2, 2);
	cv::normalize(accumMat, accumMat, 0, 255,  cv::NORM_MINMAX, CV_8UC1);

}


void LOCKYFeatureDetector_Impl::BCTS(const cv::Mat& image) {

	cv::Mat grayImage;

	assert(!image.empty());

    // Convert the image to gray scale 
    switch (image.type()) {

		case CV_32FC1:
			image.convertTo(grayImage, CV_8UC1, 255.0);
		break;

    	case CV_8UC3:
        	cvtColor(image, grayImage, CV_BGR2GRAY);
			grayImage.convertTo(grayImage, CV_8UC1);
        	break;

    	case CV_32FC3:
        	cvtColor(image, grayImage, CV_BGR2GRAY);
			grayImage.convertTo(grayImage, CV_8UC1, 255.0);
        	break;

    	default:
        	grayImage = image.clone();
        	break;
    }

	assert(grayImage.type() == CV_8UC1);
	assert(grayImage.isContinuous());

	
	accumMat = cv::Mat::zeros(grayImage.size(), CV_8UC1);

	// Init vars
	unsigned int  aux[9], weights[4], max;
	unsigned char idx;
	unsigned int  h, w, x, y;

	srand(time(0));

	// Create the integral Image
	cv::Mat integImage(grayImage.rows+1, grayImage.cols+1, CV_32FC1);
	cv::integral(grayImage, integImage);


	// Voting
	for (unsigned int it = 0; it < iter; it++) {
	
		// Step 1: create rectangles
		h = 2;
		for(int i=2; i<=(minExp + rand() % (maxExp-minExp+1)); i++) h = h * 2;
		w = 2;
		for(int i=2; i<=(minExp + rand() % (maxExp-minExp+1)); i++) w = w * 2;
		x = rand() % (integImage.cols-w-1);
		y = rand() % (integImage.rows-h-1);
		
		
        // Step 2: reduce rectangles
        for (unsigned char div = 0; div < 3; div++) {
			h = h/2;
			w = w/2;
		
			aux[0] = integImage.at<int>(y,     x);
			aux[1] = integImage.at<int>(y,     x+w);
			aux[2] = integImage.at<int>(y,     x+2*w);
			aux[3] = integImage.at<int>(y+h,   x);
			aux[4] = integImage.at<int>(y+h,   x+w);
			aux[5] = integImage.at<int>(y+h,   x+2*w);
			aux[6] = integImage.at<int>(y+2*h, x);
			aux[7] = integImage.at<int>(y+2*h, x+w);
			aux[8] = integImage.at<int>(y+2*h, x+2*w);

			weights[0] = aux[0] + aux[4] - aux[1] - aux[3];
			weights[1] = aux[1] + aux[5] - aux[2] - aux[4];
			weights[2] = aux[3] + aux[7] - aux[4] - aux[6];
			weights[3] = aux[4] + aux[8] - aux[5] - aux[7];

			max = 0;
			idx = 0;
		
			for (unsigned char i = 0; i < 4; i++) {
				if (weights[i] > max){
					max = weights[i];
					idx = i;
				}
			}
		
			switch (idx) {
				case 0:
					break;
				case 1:
					x = x + w;
					break;
				case 2:
					y = y + h;
					break;
				case 3:
					x = x + w;
					y = y + h;
					break;
				
				default:
					break;
			}
		}
		
		// Step 3: vote all pixels in final area
        for (unsigned int yAux = y; yAux <= y+h; yAux++) {
            for (unsigned int xAux = x; xAux <= x+w; xAux++) {
                accumMat.ptr<unsigned char>(yAux)[xAux] += 1;
            }
        }
	}

	// Blur the transform image and normalize it
	cv::GaussianBlur(accumMat, accumMat, cv::Size(5,5), 2, 2);
	cv::normalize(accumMat, accumMat, 0, 255,  cv::NORM_MINMAX, CV_8UC1);

}


void LOCKYFeatureDetector_Impl::extractLOCKY(std::vector<cv::KeyPoint>& keypoints) {

	keypoints.clear();

    std::vector<std::vector<cv::Point> > contours;
    std::vector<std::vector<cv::Point> > contsAux;
    std::vector<cv::Vec4i> hierarchy;
    cv::Mat binImage(accumMat.size(), CV_8UC1);
    
    cv::threshold(accumMat, binImage, thresh, UCHAR_MAX, cv::THRESH_BINARY);
    cv::findContours(binImage, contsAux, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0,0));
    
    for (std::vector<std::vector<cv::Point> >::iterator iterator=contsAux.begin(); iterator != contsAux.end(); ++iterator){
        if ((cv::contourArea(*iterator, false)) > 3)
            contours.push_back(*iterator);
    }

	for (std::vector<std::vector<cv::Point> >::iterator citer=contours.begin(); citer != contours.end(); ++citer){
        float acumX = 0;
        float acumY = 0;
        for (std::vector<cv::Point>::iterator piter=citer->begin(); piter != citer->end(); ++piter){
            acumX += piter->x;
            acumY += piter->y;
        }
        
        
        float x = acumX/(citer->size());
        float y = acumY/(citer->size());
        float s = sqrt(contourArea(*citer, false));
        
        keypoints.push_back(cv::KeyPoint(x,y,s));
        
    }
}


// Destructor
LOCKYFeatureDetector_Impl::~LOCKYFeatureDetector_Impl(void) {
	
}


//---------------------------------------------------------------------------------------------------------------------
//-----------------------------LOCKYFeatureDetector---------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------


cv::Ptr<locky::LOCKYFeatureDetector> locky::LOCKYFeatureDetector::create(int iter, int maxExp, int minExp, int thresh, bool bcts) {

    return cv::makePtr<LOCKYFeatureDetector_Impl>(iter, maxExp, minExp, thresh, bcts);
}
