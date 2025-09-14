#include <iostream>
#include <string>
#include <vector>

#include "image_io.hpp"
#include "cloud_detection.hpp"
#include "illumination_alignment.hpp"
#include "change_detection.hpp"
#include "selective_encoder.hpp"

#include <opencv2/highgui.hpp> // For imshow, waitKey (optional for debugging)


int main() {
    std::string refImagePath = "../images/reference.tif";
    std::string currentImagePath = "../images/current_capture.tif";

    std::cout << "--- EARTH+ SIMULATOR ---" << std::endl;

    
    std::cout << "\nSTEP 0: LOADING IMAGES" << std::endl;
    cv::Mat refImage = loadImageGDAL(refImagePath);
    cv::Mat currentImage = loadImageGDAL(currentImagePath);

    if (refImage.empty() || currentImage.empty()) {
        std::cerr << "FAILED TO LOAD ONE OR BOTH IMAGES. EXITING." << std::endl;
        return -1;
    }
    
    if (refImage.channels() != currentImage.channels() || refImage.depth() != currentImage.depth()) {
        std::cerr << "WARNING: REFERENCE AND CURRENT IMAGE HAVE DIFFERENT CHANNEL COUNTS OR DEPTH." << std::endl;
        
        // For simplicity, we'll proceed but this should be handled robustly
    }
    


    
    std::cout << "\nSTEP 1: CLOUD REMOVAL (SIMPLIFIED)" << std::endl;
    cv::Mat currentCloudMask = detectCloudsSimple(currentImage);
    if (currentCloudMask.empty() && !currentImage.empty()) { 
        std::cout << "CLOUD DETECTION FAILED, ASSUMING NO CLOUDS FOR CURRENT IMAGE." << std::endl;
        currentCloudMask = cv::Mat::zeros(currentImage.size(), CV_8UC1);
    }


    
    if(!currentCloudMask.empty()){
        double cloudArea = cv::sum(currentCloudMask)[0] / 255.0; 
        double totalArea = static_cast<double>(currentCloudMask.rows * currentCloudMask.cols);
        double cloudPercentage = (totalArea > 0) ? (cloudArea / totalArea) : 0.0;

        std::cout << "CLOUD PERCENTAGE IN CURRENT IMAGE: " << cloudPercentage * 100 << "%" << std::endl;
        if (cloudPercentage > 0.5) {
            std::cout << "MORE THAN 50% CLOUDY, DISCARDING CURRENT IMAGE (SIMULATION)." << std::endl;
            
        }
    }


    
    std::cout << "\nSTEP 2: ILLUMINATION ALIGNMENT" << std::endl;
    cv::Mat alignedCurrentImage = alignIllumination(currentImage, refImage, currentCloudMask);


    
    std::cout << "\nSTEP 3: CHANGE DETECTION" << std::endl;
    int tileSize = 64;
    double changeThreshold = 30.0; 
                                   
    if (alignedCurrentImage.depth() == CV_16U) changeThreshold = 700; 
    else if (alignedCurrentImage.depth() == CV_32F) changeThreshold = 0.15;


    std::vector<cv::Rect> changedTiles = detectChanges(alignedCurrentImage, refImage, changeThreshold, tileSize);


    
    #ifdef WITH_OPENCV_HIGHGUI 
    if (!changedTiles.empty() && !alignedCurrentImage.empty()) {
        cv::Mat currentImageWithChanges = alignedCurrentImage.clone();
         
        if(currentImageWithChanges.channels() == 1) {
            cv::cvtColor(currentImageWithChanges, currentImageWithChanges, cv::COLOR_GRAY2BGR);
        }
        for (const auto& rect : changedTiles) {
            cv::rectangle(currentImageWithChanges, rect, cv::Scalar(0, 0, 255), 2); 
        }
        cv::imshow("DETECTED CHANGES ON ALIGNED IMAGE", currentImageWithChanges);
        cv::waitKey(0);
        cv::destroyWindow("DETECTED CHANGES ON ALIGNED IMAGE");
    }
    #endif

    
    std::cout << "\nSTEP 4: SELECTIVE ENCODING" << std::endl;
    selectiveEncode(currentImage, changedTiles, "output_sim_earthplus.j2k");

    std::cout << "\n--- SIMULATION FINISHED ---" << std::endl;
    return 0;
}