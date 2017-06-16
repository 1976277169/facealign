//
//  LBF.h
//  myopencv
//
//  Created by lequan on 1/24/15.
//  Copyright (c) 2015 lequan. All rights reserved.
//

#ifndef FACE_ALIGNMENT_H
#define FACE_ALIGNMENT_H
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <cctype>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <opencv/cv.h>
#include <ctime>
#include <string>
#include <limits>
#include <algorithm>
#include <cmath>
#include <vector>
#include <fstream>
#include <numeric>   
#include <utility> 

#include "basic.h"

struct Params{
    
    double bagging_overlap;
    int max_numtrees;
    int max_depth;
    int landmark_num;// to be decided
    int initial_num;
    
    int max_numstage;
    double max_radio_radius[10];
    int max_numfeats[10]; // number of pixel pairs
    int max_numthreshs;
};
extern Params global_params;
extern std::string modelPath;
extern std::string dataPath;

void  TrainModel(std::string train_data_root);
double TestModel(std::string test_data_root);
int FaceDetectionAndAlignment(const char* inputname);
void ReadGlobalParamFromFile(std::string path);

#endif
