#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <opencv2/opencv.hpp>
#include "basic.h"

cv::Mat_<double> GetMeanShape(const std::vector<cv::Mat_<double> >& shapes,
	const std::vector<BoundingBox>& bounding_box);

void GetShapeResidual(const std::vector<cv::Mat_<double> >& ground_truth_shapes,
	const std::vector<cv::Mat_<double> >& current_shapes,
	const std::vector<BoundingBox>& bounding_boxs,
	const cv::Mat_<double>& mean_shape,
	std::vector<cv::Mat_<double> >& shape_residuals);

cv::Mat_<double> ProjectShape(const cv::Mat_<double>& shape, const BoundingBox& bounding_box);
cv::Mat_<double> ReProjectShape(const cv::Mat_<double>& shape, const BoundingBox& bounding_box);
void SimilarityTransform(const cv::Mat_<double>& shape1, const cv::Mat_<double>& shape2,
	cv::Mat_<double>& rotation, double& scale);
double calculate_covariance(const std::vector<double>& v_1,
	const std::vector<double>& v_2);
void LoadData(std::string filepath,
	std::vector<cv::Mat_<uchar> >& images,
	std::vector<cv::Mat_<double> >& ground_truth_shapes,
	std::vector<BoundingBox> & bounding_box);
void LoadDataAdjust(std::string filepath,
	std::vector<cv::Mat_<uchar> >& images,
	std::vector<cv::Mat_<double> >& ground_truth_shapes,
	std::vector<BoundingBox> & bounding_box);
void LoadOpencvBbxData(std::string filepath,
	std::vector<cv::Mat_<uchar> >& images,
	std::vector<cv::Mat_<double> >& ground_truth_shapes,
	std::vector<BoundingBox> & bounding_boxs
	);


BoundingBox CalculateBoundingBox(cv::Mat_<double>& shape);
cv::Mat_<double> LoadGroundtruthShape(std::string filename);
double CalculateError(const cv::Mat_<double>& ground_truth_shape, const cv::Mat_<double>& predicted_shape);

BoundingBox shape_to_bbox(const cv::Mat_<double> ground_truth_shape);

void loadTrainTestdata(std::string traindataPath, std::vector<cv::Mat_<uchar>> &faces, std::vector<cv::Mat_<double>> &shapes, std::vector<BoundingBox> &bboxes);

void resizeImg(cv::Mat &face,cv::Mat_<double> &shape,BoundingBox &bbx);

#endif

