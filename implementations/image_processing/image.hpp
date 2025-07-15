#pragma once
#include<iostream>
#include<vector>
#include<string>
#include<array>
#include<algorithm>

enum class ImageType{
    GREYSCALE,
    RGB
};


class Image{
private:
    int width, height;
    ImageType type;
    std::vector<unsigned char> greyData;
    std::vector<std::array<unsigned char, 3>>rgbData;

public:
    Image();
    Image(int width, int height, ImageType type);
    void load(int width, int height, const std::vector<unsigned char>&greyData);
    void load(int width, int height, const std::vector<std::array<unsigned char,3>>&rgbData);
    bool loadFromFile(const std::string& filename);
    bool save(const std::string &file) const;

    void invert();
    void toGreyScale();

    //getters
    int getWidth() const;
    int getHeight() const;
    ImageType getType() const;
    unsigned char getGrey(int x, int y) const;
    std::array<unsigned char, 3> getRGB(int x, int y) const;

    //setters
    void setGrey(int x, int y, unsigned char grey);
    void setRGB(int x, int y, const std::array<unsigned char, 3> &rgb);
};

