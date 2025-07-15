#pragma once
#include"image.hpp"
#include<iostream>
#include<vector>
#include<string>
#include<array>
#include<algorithm>

Image adjustBrightness(const Image &image, int amount);
Image adjustContrast(const Image &image, double factor);
Image convolve(const Image &image, const std::vector<std::vector<float>> &kernel);
Image sharpen(const Image &image);
Image detectEdges(const Image &image);
Image boxFilter(const Image &image, int ksize=3);

//Image blur(Image &image, std::pair<int,int>&point, int radius);
Image gaussianBlur(const Image &image, int size = 3, double sigma = 1.);
Image medianBlur(const Image &image, int size = 3);

