#pragma once
#include"image.hpp"
#include<iostream>
#include<vector>
#include<string>
#include<array>
#include<algorithm>

Image adjustBrightness(const Image &image, int amount);
Image adjustContrast(const Image &image, double factor);
Image convolve(const Image &image, const std::vector<std::vector<double>> &kernel);
Image sharpen(const Image &image);
Image detectEdges(const Image &image);
Image boxFilter(const Image &image, int ksize=3);

//Image blur(Image &image, std::pair<int,int>&point, int radius);
Image gaussianBlur(const Image &image, int size = 3, double sigma = 1.);
Image medianBlur(const Image &image, int size = 3);

Image threshold(Image &image);
Image dilate(Image &image);
Image erode(Image &image);
void Opening(Image &image);
void Closing(Image &image);

Image distanceTransform(const Image &image);
Image connectedComponents(const Image &binaryImage);
Image findMarkers(const Image &distanceMap, double threshold_factor = .6);
Image watershed(const Image &image, Image &markedImage);

Image instanceSegment(Image &image);