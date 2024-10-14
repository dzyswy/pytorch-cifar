#include "net.h"
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <stdio.h>
#include <vector>
#include <map>
#include <algorithm>

void print_topk(std::vector<float>& scores, int topk)
{
    std::map<int, std::string> targets_idx;
    targets_idx.insert(std::pair<int, std::string>(0, "airplane"));
    targets_idx.insert(std::pair<int, std::string>(1, "automobile"));
    targets_idx.insert(std::pair<int, std::string>(2, "bird"));
    targets_idx.insert(std::pair<int, std::string>(3, "cat"));
    targets_idx.insert(std::pair<int, std::string>(4, "deer"));
    targets_idx.insert(std::pair<int, std::string>(5, "dog"));
    targets_idx.insert(std::pair<int, std::string>(6, "frog"));
    targets_idx.insert(std::pair<int, std::string>(7, "horse"));
    targets_idx.insert(std::pair<int, std::string>(8, "ship"));
    targets_idx.insert(std::pair<int, std::string>(9, "truck"));

    std::vector<std::pair<float, int> > vec;
    vec.resize(scores.size());
    for (int i = 0; i < vec.size(); i++) {
        vec[i] = std::make_pair(scores[i], i);
    }

    std::sort(vec.begin(), vec.end(), std::greater<std::pair<float, int> >());

    for (int i = 0; (i < vec.size()) && (i < topk); i++)
    {
        printf("%d:\t score=%f,\t %s\n", i, vec[i].first, targets_idx[vec[i].second].c_str());
    }
}

int main(int argc, char* argv[])
{
    int ret;
    if (argc != 2) {
        printf("usage: %s *.png\n", argv[0]);
        return -1;
    }

    std::string img_name = argv[1];

    cv::Mat img = cv::imread(img_name);
    if (img.empty()) {
        printf("Failed to open image\n");
        return -1;
    }

    ncnn::Net cifar; 

    //ret = cifar.load_param("../model/CIFAR_10_20.param");
    ret = cifar.load_param("../model/CIFAR_10_20-int8.param");
    if (ret < 0) {
        printf("Failed to open model param\n");
        return -1;
    }

    //ret = cifar.load_model("../model/CIFAR_10_20.bin");
    ret = cifar.load_model("../model/CIFAR_10_20-int8.bin");
    if (ret < 0) {
        printf("Failed to open model bin\n");
        return -1;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows, 32, 32);

    const float mean_vals[3] = {0.5 * 255, 0.5 * 255, 0.5 * 255};
    const float std_vals[3] = {1.0 / 0.5 / 255, 1.0 / 0.5 / 255, 1.0 / 0.5 / 255};
    in.substract_mean_normalize(mean_vals, std_vals);

    ncnn::Extractor ex = cifar.create_extractor();
    ex.input("input", in);

    ncnn::Mat out;
    ex.extract("output", out);

    std::vector<float> scores;
    scores.resize(out.w);
    for (int i = 0; i < out.w; i++)
    {
        printf("%d: %f\n", i, out[i]);
        scores[i] = out[i];
    }

    print_topk(scores, 5);

    return 0;
}

