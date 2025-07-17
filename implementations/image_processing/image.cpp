#include "image.hpp"
#include<iostream>
#include<vector>
#include<array>
#include<algorithm>

#include <fstream>
#include <stdexcept>
#include <sstream>

Image::Image() : width(0), height(0), type(ImageType::GREYSCALE) {}

Image::Image(int w, int h, ImageType t) : width(w), height(h), type(t){
    if(type == ImageType::GREYSCALE){
        greyData.resize(width*height, 0);
    }
    else{
        rgbData.resize(width*height, {0,0,0});
    }
}

void Image::load(int width, int height, const std::vector<unsigned char>&greyData){
    if(greyData.size()!=width*height) throw std::invalid_argument("Size doesnt match");
    this->width = width;
    this->height = height;
    this->greyData = greyData;
    this->type = ImageType::GREYSCALE;
}

void Image::load(int width, int height, const std::vector<std::array<unsigned char, 3>>&rgbData){
    if(rgbData.size()!= width*height) throw std::invalid_argument("Size doesnt match");
    this->width = width;
    this->height = height;
    this->type = ImageType::RGB;
    this->rgbData = rgbData;
}

void Image::invert(){
    if(type == ImageType::RGB){
        for(auto& pixel: rgbData){
            pixel[0] = 255 - pixel[0];
            pixel[1] = 255 - pixel[1];
            pixel[2] = 255 - pixel[2];
        }
    }
    else{
        for(auto &pixel: greyData) pixel = 255 - pixel;
    }
}

void Image::toGreyScale(){
    if(type == ImageType::GREYSCALE) return;
    else{
        greyData.resize(width*height);
        for(int i=0;i<width*height;i++){
            const auto &rgb = rgbData[i];
            greyData[i] = static_cast<unsigned char>(0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]);
        }
        rgbData.clear();
        type = ImageType::GREYSCALE;
    }
}

int Image::getWidth() const {return width;}
int Image::getHeight() const {return height;}
ImageType Image::getType() const {return type;}

unsigned char Image::getGrey(int x, int y) const{
    if (x < 0 || y < 0 || x >= width || y >= height) throw std::out_of_range("Pixel coordinates out of bounds");
    if(type != ImageType::GREYSCALE) throw std::logic_error("types doesnt match");
    return greyData[y*width + x];
}

std::array<unsigned char, 3> Image::getRGB(int x, int y) const{
    if (x < 0 || y < 0 || x >= width || y >= height) throw std::out_of_range("Pixel coordinates out of bounds");
    if(type != ImageType::RGB) throw std::logic_error("types doesnt match");
    return rgbData[y*width+x];
}

void Image::setGrey(int x, int y, unsigned char grey){
    if (type != ImageType::GREYSCALE) throw std::logic_error("Not a grayscale image");
    if(x<0 || y<0 || x>=width|| y>=height) throw std::out_of_range("Out of Bounds error");
    greyData[y * width + x] = grey;
}

void Image::setRGB(int x, int y, const std::array<unsigned char, 3>& rgb) {
    if (type != ImageType::RGB) throw std::logic_error("Not an RGB image");
    if (x<0 || y<0 || x>=width || y>=height) throw std::out_of_range("Out of Bounds error");
    rgbData[y * width + x] = rgb;
}

bool Image::save(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) return false;

    if (type == ImageType::GREYSCALE) {
        file << "P5\n" << width << " " << height << "\n255\n";
        file.write(reinterpret_cast<const char*>(greyData.data()), greyData.size());
    } else {
        file << "P6\n" << width << " " << height << "\n255\n";
        for (const auto& pixel : rgbData) {
            file.write(reinterpret_cast<const char*>(pixel.data()), 3);
        }
    }
    return true;
}

bool Image::loadFromFile(const std::string& filename){
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    std::string line;
    std::string magic_number;
    
    std::getline(file, magic_number);
    while (file.peek() == '#') {
        std::getline(file, line);
    }
    int max_val;
    file >> width >> height >> max_val;
    file.get();

    if (magic_number == "P5" && max_val == 255) {
        type = ImageType::GREYSCALE;
        greyData.resize(width * height);
        file.read(reinterpret_cast<char*>(greyData.data()), greyData.size());
    } else if (magic_number == "P6" && max_val == 255) {
        type = ImageType::RGB;
        rgbData.resize(width * height);
        file.read(reinterpret_cast<char*>(rgbData.data()), rgbData.size() * 3);
    } else {
        return false;
    }
    return true;
}

std::vector<int> Image::create_histogram(){
    std::vector<int>histogram(256,0);
    if(type==ImageType::RGB) throw std::logic_error("Histograms are just for greyScale for now");
    for(int i = 0; i < width; i++){
        for(int j = 0; j < height; j++){
            histogram[getGrey(i,j)]++;
        }
    }
    return histogram;
}

