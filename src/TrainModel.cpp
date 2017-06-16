//
//  TrainDemo.cpp
//  myopencv
//
//  Created by lequan on 1/24/15.
//  Copyright (c) 2015 lequan. All rights reserved.
//
#include "LBFRegressor.h"
#include "utils.h"
using namespace std;
using namespace cv;

void TrainModel(string traindata_path){
	vector<Mat_<uchar>> faces;
    vector<Mat_<double>> ground_truth_shapes;
    vector<BoundingBox> bounding_boxs;
	
	//load traindata,get the bbox to obtain a face crop 
	loadTrainTestdata(traindata_path, faces, ground_truth_shapes, bounding_boxs);
    LBFRegressor regressor;
	regressor.Train(faces, ground_truth_shapes, bounding_boxs);
    regressor.Save(modelPath+"LBF.model");
    return;
}


