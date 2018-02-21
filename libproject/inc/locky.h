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

#ifndef _LOCKY_H_
#define _LOCKY_H_


#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>


namespace locky{


	class CV_EXPORTS_W LOCKYFeatureDetector : public cv::Feature2D {
	public:

		static cv::Ptr<LOCKYFeatureDetector> create(int iter = 100000, int maxExp = 7, int minExp = 3, int thresh = 30, bool bcts = true);

		virtual void getAccumMat(cv::Mat& toReturn) = 0;

		virtual void detect(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints) = 0;

	};

}

#endif // _LOCKY_H_
