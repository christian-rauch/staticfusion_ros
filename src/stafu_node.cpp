#include <StaticFusion.h>

#include <iostream>
#include <fstream>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include <image_transport/subscriber_filter.h>

struct sf_conf_t {
    int im_count;
    unsigned int res_factor;
    cv::Mat weightedImage;
    cv::Mat depth_full;
    cv::Mat color_full;

    bool denseModel;
    bool modelInitialised;
};

void callback(const sensor_msgs::ImageConstPtr& msg_colour, const sensor_msgs::ImageConstPtr& msg_depth, StaticFusion &staticFusion, sf_conf_t &conf) {

    // decode images
    conf.color_full = cv_bridge::toCvShare(msg_colour)->image;  // 8bit RGB image
    conf.depth_full = cv_bridge::toCvShare(msg_depth)->image;   // 16bit depth image

    if(conf.color_full.size()!=conf.depth_full.size())
        throw std::runtime_error("resolution mismatch");

    // res_factor: (1 - 640 x 480, 2 - 320 x 240)
    cv::Size target;
    switch (conf.res_factor) {
    case 1: target = cv::Size(640, 480); break;
    case 2: target = cv::Size(320, 240); break;
    default:
        throw std::runtime_error("unsupported camera resolution mode");
    }

    const cv::Size current_size = conf.color_full.size();
    if(current_size!=target) {
        // nearest neighbour interpolation
        cv::resize(conf.color_full, conf.color_full, target, 0, 0, cv::INTER_NEAREST);
        cv::resize(conf.depth_full, conf.depth_full, target, 0, 0, cv::INTER_NEAREST);
    }

    if(conf.depth_full.type()!=CV_16UC1) {
        throw std::runtime_error("expect depth in 16bit mm values ¯\\_(ツ)_/¯");
    }

    // flip horizontally
    cv::flip(conf.color_full, conf.color_full, 0);
    cv::flip(conf.depth_full, conf.depth_full, 0);

    std::vector<cv::Mat> rgb(3); // colour channels
    cv::split(conf.color_full, rgb);//split source

    // convert images
    const cv::Mat_<float> grey = (0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]) / 255;
    cv::cv2eigen(grey, staticFusion.intensityCurrent);
    cv::cv2eigen(conf.depth_full/1000, staticFusion.depthCurrent);

    // initialise
    if (conf.im_count == 0) {
        staticFusion.kb = 1.05f;

        staticFusion.intensityPrediction = staticFusion.intensityCurrent;
        staticFusion.depthPrediction = staticFusion.depthCurrent;

        staticFusion.depthBuffer[conf.im_count % staticFusion.bufferLength] = staticFusion.depthPrediction.replicate(1,1);
        staticFusion.intensityBuffer[conf.im_count % staticFusion.bufferLength] = staticFusion.intensityPrediction.replicate(1,1);
        staticFusion.odomBuffer[conf.im_count % staticFusion.bufferLength] = Eigen::Matrix4f::Identity();

        staticFusion.createImagePyramid(true);

        staticFusion.runSolver(true);

        staticFusion.buildSegmImage();

        conf.im_count++;

        staticFusion.depthBuffer[conf.im_count % staticFusion.bufferLength] = staticFusion.depthCurrent.replicate(1,1);
        staticFusion.intensityBuffer[conf.im_count % staticFusion.bufferLength] = staticFusion.intensityCurrent.replicate(1,1);
        staticFusion.odomBuffer[conf.im_count % staticFusion.bufferLength] = staticFusion.T_odometry;

        cv::eigen2cv(staticFusion.b_segm_perpixel, conf.weightedImage);

        staticFusion.reconstruction->fuseFrame((unsigned char *) conf.color_full.data, (unsigned short *) conf.depth_full.data, (float *) conf.weightedImage.data, conf.im_count, &(staticFusion.T_odometry), 0, 1);
        staticFusion.reconstruction->uploadWeightAndClustersForVisualization((float *) conf.weightedImage.data, staticFusion.clusterAllocation[0], (unsigned short *) conf.depth_full.data);
    }

    conf.im_count++;

    conf.denseModel = staticFusion.reconstruction->checkIfDenseEnough();

    if (!conf.denseModel && !conf.modelInitialised) {

        staticFusion.kb = 1.05f; //1.25s
        conf.modelInitialised = true;

    } else {

        staticFusion.kb = 1.5f;
        conf.modelInitialised = true;
    }

    staticFusion.reconstruction->getPredictedImages(staticFusion.depthPrediction, staticFusion.intensityPrediction);
    staticFusion.reconstruction->getFilteredDepth(conf.depth_full, staticFusion.depthCurrent);

    staticFusion.createImagePyramid(true);   //pyramid for the old model

    staticFusion.runSolver(true);

    if (conf.im_count - staticFusion.bufferLength >= 0) {
        staticFusion.computeResidualsAgainstPreviousImage(conf.im_count);
    }

    //Build segmentation image to use it for the data fusion
    staticFusion.buildSegmImage();

    staticFusion.depthBuffer[conf.im_count % staticFusion.bufferLength] = staticFusion.depthCurrent.replicate(1,1);
    staticFusion.intensityBuffer[conf.im_count % staticFusion.bufferLength] = staticFusion.intensityCurrent.replicate(1,1);
    staticFusion.odomBuffer[conf.im_count % staticFusion.bufferLength] = staticFusion.T_odometry;

    cv::eigen2cv(staticFusion.b_segm_perpixel, conf.weightedImage);

    staticFusion.reconstruction->fuseFrame((unsigned char *) conf.color_full.data, (unsigned short *) conf.depth_full.data,  (float *) conf.weightedImage.data, conf.im_count, &(staticFusion.T_odometry), 0, 1);

    staticFusion.reconstruction->uploadWeightAndClustersForVisualization((float *) conf.weightedImage.data, staticFusion.clusterAllocation[0], (unsigned short *) conf.depth_full.data);
}

int main(int argc, char** argv) {

    unsigned int res_factor = 2;
    StaticFusion staticFusion(res_factor);

    //Flags
    staticFusion.use_motion_filter = true;

    //Solver
    staticFusion.ctf_levels = log2(staticFusion.cols/40) + 2;
    staticFusion.max_iter_per_level = 3;
    staticFusion.previous_speed_const_weight = 0.1f;
    staticFusion.previous_speed_eig_weight = 2.f; //0.5f;

    staticFusion.k_photometric_res = 0.15f;
    staticFusion.irls_delta_threshold = 0.0015f;
    staticFusion.max_iter_irls = 6;
    staticFusion.lambda_reg = 0.35f; //0.4
    staticFusion.lambda_prior = 0.5f; //0.5
    staticFusion.kc_Cauchy = 0.5f; //0.5
    staticFusion.kb = 1.5f; //1.5
    staticFusion.kz = 1.5f;

    sf_conf_t sf_conf;

    sf_conf.weightedImage = cv::Mat(Resolution::getInstance().width(), Resolution::getInstance().height(), CV_32F, 0.0);

    sf_conf.depth_full = cv::Mat(staticFusion.height, staticFusion.width,  CV_16U, 0.0);
    sf_conf.color_full = cv::Mat(staticFusion.height, staticFusion.width,  CV_8UC3,  cv::Scalar(0,0,0));

    sf_conf.denseModel = false;
    sf_conf.modelInitialised = false;

    sf_conf.im_count = 0;

    sf_conf.res_factor = res_factor;

    // init ROS
    ros::init(argc, argv, "stafu");
    ros::NodeHandle n;
    image_transport::ImageTransport it(n);

    image_transport::SubscriberFilter sub_colour(it, "colour", 1);
    image_transport::SubscriberFilter sub_depth(it, "depth", 1);

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> ApproximateTimePolicy;
    message_filters::Synchronizer<ApproximateTimePolicy> sync(ApproximateTimePolicy(5), sub_colour, sub_depth);

    sync.registerCallback(boost::bind(&callback, _1, _2, staticFusion, sf_conf));

    while(!pangolin::ShouldQuit()) {
        ros::spinOnce();
        staticFusion.updateGUI();
    }

    return EXIT_SUCCESS;
}
