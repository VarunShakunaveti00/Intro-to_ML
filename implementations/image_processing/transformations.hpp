#pragma once
#include"image.hpp"
#include<iostream>
#include<vector>
#include<string>
#include<array>
#include<algorithm>


Image crop(const Image &image, const std::pair<int, int>&topLeft, const std::pair<int, int>&bottomRight);

void flipHorizontal(Image &image);
void flipVertical(Image &image);
void resize(Image &image, int newWidth, int newHeight);

Image rotate(Image &image, double angle);

